"""Futbol 2v2 con HUMANOIDES de dm_control.locomotion.soccer (CMU humanoid, 56 articulaciones).

IMPORTANTE (honestidad): equilibrarse y caminar con 56 servos requiere una politica entrenada con RL.
DeepMind no publica la del futbol. Para poder jugar YA, este script usa CAMINATA ASISTIDA:
  * un "arnes" virtual (fuerzas externas sobre la pelvis) mantiene al humanoide parado y lo desplaza;
  * las piernas y los brazos hacen una animacion de marcha / patada con los servos de posicion.
Es util para aprender el entorno y generar videos. NO es locomocion aprendida: ese es el siguiente paso.

Instalar:  pip install dm_control imageio imageio-ffmpeg pillow
Uso:
  python humanoid_soccer.py --seed 1 --seconds 10     # prueba rapida
  python humanoid_soccer.py                           # busca una semilla con goles
Nota: el entorno original descarga un archivo de mocap de Google; aca usamos un inicializador
simple (UprightInitializer) para no depender de ninguna descarga.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, time
import numpy as np, mujoco
from dm_control import composer, mjcf
from dm_control.locomotion import soccer as S
from dm_control.locomotion.soccer.humanoid import Humanoid
from dm_control.locomotion.soccer.pitch import (MINI_FOOTBALL_GOAL_SIZE, MINI_FOOTBALL_MAX_AREA_PER_HUMANOID,
                                                MINI_FOOTBALL_MIN_AREA_PER_HUMANOID, RandomizedPitch)
from dm_control.locomotion.soccer.soccer_ball import regulation_soccer_ball
from dm_control.locomotion.soccer.task import MultiturnTask
from dm_control.locomotion.soccer.team import Player, Team, RGBA_BLUE, RGBA_RED
from dm_control.locomotion.walkers.initializers import UprightInitializer
from dm_control.rl.control import PhysicsError

ATT_SPEED, SUP_SPEED, KICK_TIME = 3.5, 3.0, 0.5     # velocidades de carrera (m/s) y duracion de la patada (s)

# ----------------------------------------------------------------------------- entorno
def make_env(seed, seconds, team_size=2):
    players = []
    for team, rgba, name in ((Team.HOME, RGBA_BLUE, "home"), (Team.AWAY, RGBA_RED, "away")):
        for i in range(team_size):
            w = Humanoid(name=f"{name}{i}", marker_rgba=list(rgba), walker_id=i,
                         visual=Humanoid.Visual.JERSEY, initializer=UprightInitializer())
            players.append(Player(team, w))
    n = team_size * 2
    arena = RandomizedPitch(min_size=S._area_to_size(MINI_FOOTBALL_MIN_AREA_PER_HUMANOID * n),
                            max_size=S._area_to_size(MINI_FOOTBALL_MAX_AREA_PER_HUMANOID * n),
                            keep_aspect_ratio=False, field_box=False, goal_size=MINI_FOOTBALL_GOAL_SIZE)
    return composer.Environment(
        task=MultiturnTask(players=players, arena=arena, ball=regulation_soccer_ball(),
                           disable_walker_contacts=True),        # el humanoide-titere no debe chocar con otros
        time_limit=seconds, random_state=seed)

# ----------------------------------------------------------------------------- jugador asistido
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-6 else v

class Assisted:
    """Control de un humanoide: arnes en la pelvis + marcha animada con los servos."""
    def __init__(self, env, idx, prefix):
        self.p = env.physics; self.prefix = prefix; self.idx = idx
        walker = env.task.players[idx].walker
        self.rj = self.p.bind(mjcf.get_attachment_frame(walker.mjcf_model).freejoint)   # freejoint de la pelvis
        self.slots = {}                                    # nombre corto -> (indice de accion, lo, hi)
        m = self.p.model
        for k, act in enumerate(walker.actuators):
            full = act.full_identifier
            j = m.actuator_trnid[m.name2id(full, "actuator"), 0]
            lo, hi = m.jnt_range[j]
            self.slots[full.split("/")[1]] = (k, lo, hi)
        self.mass = float(self.p.named.model.body_subtreemass[prefix + "root"])
        self.phase = 1.7 * idx                           # determinista: misma semilla -> mismo partido
        self.kick_t, self.cool = -1.0, 0.0
        self.z0 = float(self.p.named.data.xpos[prefix + "root"][2])
        self.field = tuple(np.array(env.task.arena.size)[:2] - 0.5)       # limites de la cancha

    def pos(self): return self.p.named.data.xpos[self.prefix + "root"].copy()
    def yaw(self):
        R = self.p.named.data.xmat[self.prefix + "root"].reshape(3, 3)
        return np.arctan2(R[1, 2], R[0, 2])          # en el CMU humanoid: eje local z = adelante

    def apply_harness(self, yaw_des, speed, dt=0.025):
        """Pelvis 'titere': orientacion siempre erguida, rumbo y velocidad impuestos (con limites de
        giro y aceleracion). Piernas, brazos, tronco y contactos con la pelota siguen siendo fisica real."""
        d, nm = self.p.named.data, self.prefix + "root"
        q = np.array(self.rj.qpos, dtype=float).reshape(-1)   # [x y z qw qx qy qz]
        # rumbo con velocidad de giro limitada
        self.yaw_s = getattr(self, "yaw_s", self.yaw())
        self.yaw_s = wrap(self.yaw_s + np.clip(wrap(yaw_des - self.yaw_s), -3.5 * dt, 3.5 * dt))
        # velocidad con aceleracion limitada
        v = getattr(self, "vel", np.zeros(2))
        vdes = speed * np.array([np.cos(self.yaw_s), np.sin(self.yaw_s)])
        v = v + np.clip(vdes - v, -6.0 * dt, 6.0 * dt)
        self.vel = v
        pos = q[:2] + v * dt
        hx, hy = self.field
        pos = np.clip(pos, [-hx, -hy], [hx, hy])
        c, s_ = np.cos(self.yaw_s), np.sin(self.yaw_s)
        R = np.column_stack([[-s_, c, 0], [0, 0, 1], [c, s_, 0]])                  # columnas: izquierda, arriba, adelante
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, R.flatten())
        q[:2] = pos; q[2] = self.z0; q[3:7] = quat
        self.rj.qpos = q
        self.rj.qvel = np.array([v[0], v[1], 0, 0, 0, 0])

    def action(self, speed, t, kick):
        """Vector de 56 acciones normalizadas [-1,1] -> posiciones objetivo de cada articulacion."""
        a = np.zeros(56)
        kick = -1.0 if kick is None else kick
        def setq(name, q):
            if name in self.slots:
                k, lo, hi = self.slots[name]
                a[k] = np.clip(2 * (np.clip(q, lo, hi) - lo) / (hi - lo) - 1, -1, 1)
        for name in self.slots:                                   # pose neutra (q=0) para todas las articulaciones
            setq(name, 0.0)
        amp = min(1.0, speed / 3.0)
        self.phase += 0.025 * 2 * np.pi * (1.2 + 1.2 * amp) * (1 if speed > 0.1 else 0)
        s, c = np.sin(self.phase), np.cos(self.phase)
        for side, sg in (("l", 1), ("r", -1)):
            hip = -0.65 * amp * sg * s                            # flexion = negativo
            knee = 0.15 + 1.0 * amp * max(0.0, -sg * c) ** 1.0 if True else 0
            setq(side + "femurrx", hip)
            setq(side + "tibiarx", knee)
            setq(side + "footrx", -0.2 * hip)
            setq(side + "humerusrx", 0.0); setq(side + "humerusry", 0.0)
            setq(side + "humerusrz", sg * 1.3)                    # bajar los brazos
            setq(side + "radiusrx", 0.5)
            setq(side + "humerusrx", 0.5 * amp * sg * s)          # braceo opuesto a la pierna
        if kick >= 0:                                             # patada con la pierna derecha
            hip = np.interp(kick, [0, .35, .65, 1], [0.0, 0.6, -0.9, -1.1])    # atras -> barrido -> seguimiento
            knee = np.interp(kick, [0, .35, .65, 1], [0.3, 1.4, 0.2, 0.4])
            setq("rfemurrx", hip); setq("rtibiarx", knee); setq("rfootrx", 0.3)
        return a

# ----------------------------------------------------------------------------- decisiones (coords mundo)
class Team2v2:
    def __init__(self, env):
        self.env = env; self.p = env.physics
        names = ["home0", "home1", "away0", "away1"]
        self.bots = [Assisted(env, i, n + "/") for i, n in enumerate(names)]
        self.kick = [-1.0] * 4

    def ball(self): return self.p.named.data.xpos["soccer_ball/"][:2].copy()
    def goal(self, team):                         # arco que ataca el equipo 'team' (0=home, 1=away)
        return self.p.named.data.geom_xpos["away_goal/top_post" if team == 0 else "home_goal/top_post"][:2].copy()

    def step(self, ts, dt=0.025):
        ball = self.ball(); acts = []
        pos = [b.pos()[:2] for b in self.bots]
        for team, ids in ((0, (0, 1)), (1, (2, 3))):
            goal = self.goal(team)
            dist = [np.linalg.norm(ball - pos[i]) for i in ids]
            att = ids[int(np.argmin(dist))]
            for i in ids:
                b = self.bots[i]; me = pos[i]; sd = unit(goal - ball)
                if i == att:
                    d = np.linalg.norm(ball - me)
                    ahead = np.dot(unit(ball - me), sd) > 0.3
                    perp = np.array([-sd[1], sd[0]])                      # "izquierda" de la linea pelota->arco
                    # la pelota debe quedar alineada con el PIE DERECHO (10 cm a la derecha de la pelvis)
                    target = goal if (d < 1.4 and ahead) else ball - 1.0 * sd + 0.12 * perp
                    speed = ATT_SPEED if d > 0.6 else 0.8 * ATT_SPEED
                else:                                                       # apoyo
                    perp = np.array([-sd[1], sd[0]])
                    side = 1.0 if np.dot(perp, me - ball) >= 0 else -1.0
                    target = ball - 1.5 * sd + side * 3.0 * perp
                    speed = SUP_SPEED
                to = target - me; dd = np.linalg.norm(to)
                yaw_des = np.arctan2(to[1], to[0]) if dd > 0.15 else b.yaw()
                sp = speed * np.clip(dd / 1.0, 0, 1) * max(0.0, np.cos(wrap(yaw_des - b.yaw()))) ** 2
                # patada si estoy cerca de la pelota, la tengo delante y es el atacante
                to_ball = np.arctan2(*(ball - me)[::-1]); to_goal = np.arctan2(*(goal - ball)[::-1])
                if i == att and np.linalg.norm(ball - me) < 1.2 and self.kick[i] < 0 and b.cool <= 0 \
                        and abs(wrap(to_ball - b.yaw())) < 0.6 and abs(wrap(to_goal - b.yaw())) < 0.5:
                    self.kick[i] = 0.0                   # patear solo si la pelota esta delante Y el arco en esa linea
                if self.kick[i] >= 0:
                    self.kick[i] += dt / KICK_TIME; b.cool = 0.8
                    if self.kick[i] > 1: self.kick[i] = -1.0
                b.cool = max(0.0, b.cool - dt)
                if self.kick[i] >= 0: sp = min(sp, 0.6)          # frenar mientras patea
                b.apply_harness(yaw_des, sp)
                acts.append(b.action(sp, 0.0, self.kick[i]))
        return acts

# ----------------------------------------------------------------------------- partido, video
def goal_of(ts): return int(np.sign(float(ts.reward[0])))

def new_match(seed, seconds):
    env = make_env(seed, seconds); ts = env.reset()
    env.physics.model.light_castshadow[:] = 0                 # sombras desactivadas (mucho mas rapido sin GPU)
    return env, ts, Team2v2(env)

def play(seed, seconds):
    """Devuelve (goles_azul, goles_rojo), o None si la fisica se inestabiliza en esa semilla."""
    env, ts, team = new_match(seed, seconds); home = away = 0
    try:
        while not ts.last():
            ts = env.step(team.step(ts)); g = goal_of(ts)
            home += g > 0; away += g < 0
    except PhysicsError:
        return None
    return home, away

def find_seed(seconds, candidates=10):
    best, key_best = 1, (-1, -1)
    for sd in range(1, candidates + 1):
        res = play(sd, seconds)
        if res is None: print(f"  semilla {sd}: fisica inestable, se descarta", flush=True); continue
        h, a = res; print(f"  semilla {sd}: AZUL {h} - {a} ROJO", flush=True)
        key = (h + a, min(h, a))
        if key > key_best: best, key_best = sd, key
    return best

def overlay(img, text, big=None):
    from PIL import Image, ImageDraw, ImageFont
    H, W = img.shape[:2]; im = Image.fromarray(img); dr = ImageDraw.Draw(im); k = W / 1280
    def font(sz):
        for p in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"):
            try: return ImageFont.truetype(p, sz)
            except OSError: pass
        return ImageFont.load_default(size=sz)
    dr.rectangle([W//2 - 190*k, 12*k, W//2 + 190*k, 70*k], fill=(0, 0, 0))
    dr.text((W//2, 41*k), text, font=font(int(36*k)), anchor="mm", fill=(255, 255, 255))
    if big:
        sz = int((150 + 12 * np.sin(big * 12)) * k)
        dr.text((W//2, int(H*.3)), "GOOOOL!", font=font(sz), anchor="mm", fill=(255, 214, 0),
                stroke_width=max(2, int(8*k)), stroke_fill=(20, 20, 20))
    return np.asarray(im)

def add_minimap(img, mini):
    h, w = mini.shape[:2]; m = 10; img = img.copy()
    img[-h - m - 2:-m + 2, m - 2:m + w + 2] = 255; img[-h - m:-m, m:m + w] = mini
    return img

def make_video(seed, seconds, path, camera, width=640, minimap=False, every=2):
    import imageio
    W, H = width, width * 9 // 16
    env, ts, team = new_match(seed, seconds)
    env.physics.model.vis.global_.offwidth = W; env.physics.model.vis.global_.offheight = H
    dt = env.control_timestep(); fps = round(1 / dt / every); total = int(seconds / dt / every)
    k = n = 0; t = 0.0; last_goal_t = -99.0; home = away = 0; t0 = time.time()
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        while not ts.last():
            try:
                ts = env.step(team.step(ts))
            except PhysicsError:
                print("  la fisica se inestabilizo: el video termina aca (proba otra --seed)"); break
            t += dt; k += 1
            g = goal_of(ts)
            if g: home += g > 0; away += g < 0; last_goal_t = t
            if k % every: continue
            img = env.physics.render(H, W, camera_id=camera)
            if minimap: img = add_minimap(img, env.physics.render(H // 4, W // 4, camera_id="top_down"))
            since = t - last_goal_t
            w.append_data(overlay(img, f"AZUL {home} - {away} ROJO", big=since if since < 2.5 else None))
            n += 1
            if n == 1 or n % 20 == 0:
                spf = (time.time() - t0) / n
                print(f"  frame {n}/{total} | {spf:.2f} s/frame | faltan ~{(total - n) * spf:.0f} s", flush=True)
    print(f"Video: {path} | semilla {seed} | marcador final AZUL {home} - {away} ROJO")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="auto"); ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--camera", default="soccer_ball/ball_cam", help="soccer_ball/ball_cam | ball_cam_near | ball_cam_far | top_down | home0/float_far")
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--minimap", action="store_true"); ap.add_argument("--out", default="humanoid_soccer.mp4")
    a = ap.parse_args()
    seed = find_seed(a.seconds) if a.seed == "auto" else int(a.seed)
    make_video(seed, a.seconds, a.out, a.camera, a.width, a.minimap, a.every)