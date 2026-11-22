"""Futbol 2v2 con dm_control.locomotion.soccer (DeepMind) + politica scripteada.

Instalar:  pip install dm_control imageio imageio-ffmpeg pillow
Uso:
  python dm_soccer_2v2.py                 # busca un partido con goles y genera soccer_2v2.mp4
  python dm_soccer_2v2.py --seed 3 --seconds 40   # semilla fija
  python dm_soccer_2v2.py --view          # ademas abre el viewer interactivo
En WSL, si falla el render:  MUJOCO_GL=egl python dm_soccer_2v2.py

Que es: el entorno trae cancha, arcos, pelota y 4 jugadores (BOXHEAD: cuerpo simple con
3 acciones: avanzar/retroceder, girar y saltar). NO trae politica: aca escribimos reglas
a mano (el mas cercano a la pelota ataca, el otro apoya por el costado). Despues se podria reemplazar
la funcion `policy` por una red neuronal entrenada con RL.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, numpy as np
from dm_control.locomotion import soccer

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-6 else v

def drive(target, arrive=3.0):
    """Ir a un punto dado en el marco del robot (x adelante, y izquierda).
    Convencion de BOXHEAD: accion[0] = -1 es avanzar, accion[1] = -1 gira hacia la izquierda."""
    ang = np.arctan2(target[1], target[0])
    dist = np.linalg.norm(target[:2])
    # girar casi en el lugar si el objetivo esta muy de costado; avanzar cuando ya esta de frente
    speed = np.clip(dist / arrive, 0, 1) * max(0.0, np.cos(ang)) ** 3
    return np.array([-speed, np.clip(-1.2 * ang, -1, 1), 0.0])

def policy(obs, role):
    """obs: observacion del jugador (marco egocentrico). role: 'attack' | 'support'."""
    ball = obs["ball_ego_position"][0][:2]
    goal = obs["opponent_goal_mid"][0][:2]
    shoot_dir = unit(goal - ball)
    if role == "support":
        # acompanar la jugada abierto por el costado, un poco detras de la pelota
        perp = np.array([-shoot_dir[1], shoot_dir[0]])
        side = 1.0 if np.dot(perp, -ball) >= 0 else -1.0     # quedarse del lado en que ya esta
        return drive(ball - 2.0 * shoot_dir + side * 6.0 * perp)
    # atacar: si la pelota esta cerca y delante, conducirla hacia el arco rival;
    # si no, ir a un punto justo DETRAS de la pelota (respecto del arco rival)
    if np.linalg.norm(ball) < 3.0 and np.dot(unit(ball), shoot_dir) > 0.3:
        return drive(goal, arrive=1.0)
    return drive(ball - 1.5 * shoot_dir)

def assign_roles(obs_team):
    """Entre los 2 de un equipo, ataca el mas cercano a la pelota; el otro apoya."""
    d0 = np.linalg.norm(obs_team[0]["ball_ego_position"][0][:2])
    d1 = np.linalg.norm(obs_team[1]["ball_ego_position"][0][:2])
    return ("attack", "support") if d0 <= d1 else ("support", "attack")

def actions_for(ts):
    obs = ts.observation
    acts = [None] * 4
    for team in ((0, 1), (2, 3)):          # (home0, home1), (away0, away1)
        roles = assign_roles([obs[i] for i in team])
        for i, r in zip(team, roles):
            acts[i] = policy(obs[i], r)
    return acts

def goal_of(ts):
    """+1 si anoto AZUL (home), -1 si anoto ROJO (away), 0 si no hubo gol en este paso."""
    r = float(ts.reward[0]); return int(np.sign(r))

def make_env(seed, seconds):
    return soccer.load(team_size=2, time_limit=seconds, random_state=seed,
                       walker_type=soccer.WalkerType.BOXHEAD, terminate_on_goal=False)

def overlay(img, text, big=None):
    from PIL import Image, ImageDraw, ImageFont
    H, W = img.shape[:2]
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    def font(sz):
        for p in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"):
            try: return ImageFont.truetype(p, sz)
            except OSError: pass
        return ImageFont.load_default(size=sz)
    k = W / 1280                                   # escala segun resolucion
    dr.rectangle([W//2 - 190*k, 12*k, W//2 + 190*k, 70*k], fill=(0, 0, 0))
    dr.text((W//2, 41*k), text, font=font(int(36*k)), anchor="mm", fill=(255, 255, 255))
    if big:
        sz = int((150 + 12 * np.sin(big * 12)) * k)
        dr.text((W//2, int(H*.3)), "GOOOOL!", font=font(sz), anchor="mm",
                fill=(255, 214, 0), stroke_width=max(2, int(8*k)), stroke_fill=(20, 20, 20))
    return np.asarray(im)

def add_minimap(img, mini):
    """Pega el minimapa (vista cenital) en la esquina inferior izquierda con un marco blanco."""
    h, w = mini.shape[:2]; m = 10
    img = img.copy()
    img[-h - m - 2:-m + 2, m - 2:m + w + 2] = 255
    img[-h - m:-m, m:m + w] = mini
    return img

def play(seed, seconds, on_step=None):
    """Simula un partido completo. Devuelve (goles_azul, goles_rojo)."""
    env = make_env(seed, seconds)
    ts = env.reset()
    home = away = 0
    while not ts.last():
        ts = env.step(actions_for(ts))
        g = goal_of(ts)
        home += g > 0; away += g < 0
        if on_step: on_step(env, ts, g, home, away)
    return home, away

def find_seed(seconds, candidates=12):
    """Busca una semilla con partido movido (mas goles, ideal de ambos equipos)."""
    best, best_key = 1, (-1, -1)
    for sd in range(1, candidates + 1):
        h, a = play(sd, seconds)
        key = (h + a, min(h, a))
        print(f"  semilla {sd}: AZUL {h} - {a} ROJO")
        if key > best_key: best, best_key = sd, key
    return best

def make_video(seed, seconds, path, camera, width=640, minimap=False, every=2, shadows=False):
    """every=2 -> se renderiza 1 de cada 2 pasos (video a 20 fps, igual de largo en tiempo real)."""
    import imageio, time
    W, H = width, width * 9 // 16
    env = make_env(seed, seconds); ts = env.reset()
    env.physics.model.vis.global_.offwidth = W      # el buffer por defecto es 640x480
    env.physics.model.vis.global_.offheight = H
    if not shadows:
        env.physics.model.light_castshadow[:] = 0   # sombras = ~25x mas lento en render por software
    dt = env.control_timestep(); fps = round(1 / dt / every)
    total = int(seconds / dt / every)
    k, n, t, last_goal_t, home, away = 0, 0, 0.0, -99.0, 0, 0
    t0 = time.time()
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        while not ts.last():
            ts = env.step(actions_for(ts)); t += dt; k += 1
            g = goal_of(ts)
            if g: home += g > 0; away += g < 0; last_goal_t = t
            if k % every: continue                  # simular todos los pasos, renderizar algunos
            img = env.physics.render(H, W, camera_id=camera)
            if minimap:
                img = add_minimap(img, env.physics.render(H // 4, W // 4, camera_id="top_down"))
            since = t - last_goal_t
            w.append_data(overlay(img, f"AZUL {home} - {away} ROJO", big=since if since < 2.5 else None))
            n += 1
            if n == 1 or n % 20 == 0:
                spf = (time.time() - t0) / n
                print(f"  frame {n}/{total} | {spf:.2f} s/frame | faltan ~{(total - n) * spf:.0f} s", flush=True)
                if n == 1 and spf > 0.5:
                    print("  (render lento: probablemente por software, sin GPU. Baja --width o sube --every)")
    print(f"Video: {path} | semilla {seed} | marcador final AZUL {home} - {away} ROJO")

def run_viewer(seed, seconds):
    from dm_control import viewer
    viewer.launch(make_env(seed, seconds), policy=lambda ts: actions_for(ts))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="auto", help="numero, o 'auto' para buscar un partido con goles")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--camera", default="soccer_ball/ball_cam_far",
                    help="soccer_ball/ball_cam_far | top_down | home0/float_far | away0/float_far")
    ap.add_argument("--width", type=int, default=640, help="ancho del video (alto = 16:9)")
    ap.add_argument("--every", type=int, default=2, help="renderizar 1 de cada N pasos (mas alto = mas rapido)")
    ap.add_argument("--minimap", action="store_true", help="agregar minimapa (el doble de lento)")
    ap.add_argument("--shadows", action="store_true", help="activar sombras (mucho mas lento sin GPU)")
    ap.add_argument("--out", default="soccer_2v2.mp4")
    ap.add_argument("--view", action="store_true", help="abrir tambien el viewer interactivo")
    a = ap.parse_args()
    seed = find_seed(a.seconds) if a.seed == "auto" else int(a.seed)
    try:
        make_video(seed, a.seconds, a.out, a.camera, a.width, a.minimap, a.every, a.shadows)
    except Exception as e:
        print("No se pudo generar el video:", repr(e))
        print("Proba: pip install imageio imageio-ffmpeg pillow  (o MUJOCO_GL=egl python dm_soccer_2v2.py)")
    if a.view:
        run_viewer(seed, a.seconds)