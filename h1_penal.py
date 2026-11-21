"""Penal 1v1 con dos Unitree H1 (Menagerie) usando MjSpec.
Etapa 1: bases fijas (sin equilibrio) -> pateador con PD por torque + arquero parado.

Instalar:  pip install mujoco mujoco-menagerie imageio imageio-ffmpeg pillow
Uso:
  python h1_penal.py               # genera penal_h1.mp4 y despues abre el viewer
  python h1_penal.py --no-viewer   # solo el video
  python h1_penal.py --no-video    # solo el viewer
"""
import sys, time, numpy as np, mujoco
import mujoco_menagerie as mm

# ---------------- Parametros ajustables ----------------
KICK_YAW  = 0.0      # grados: hacia donde apunta el pateador (+ = izquierda mirando al arco)
BALL_POS  = (-0.60, -0.20)   # (x, y) de la pelota
STRIKE_KNEE = 0.0    # rodilla en el golpe
GOAL_X, GOAL_HW, GOAL_H, GOAL_D = 6.5, 2.0, 2.44, 1.5   # linea de gol, semiancho, alto, profundidad
FPS, W, H = 60, 1280, 720
VIDEO = "penal_h1.mp4"
SIM_TIME = 5.0
# --------------------------------------------------------

CAP, BOX = mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX
def visual(**kw):            # geometria que se ve pero no colisiona
    return dict(contype=0, conaffinity=0, **kw)

def add_goal(world):
    wb, white = world.worldbody, [1, 1, 1, 1]
    x0, x1, hw, h = GOAL_X, GOAL_X + GOAL_D, GOAL_HW, GOAL_H
    # Marco (postes + travesano + estructura trasera)
    for y in (-hw, hw):
        wb.add_geom(type=CAP, size=[.06, 0, 0], fromto=[x0, y, 0, x0, y, h], rgba=white)
        wb.add_geom(type=CAP, size=[.04, 0, 0], fromto=[x1, y, 0, x1, y, h], rgba=white, **visual())
        wb.add_geom(type=CAP, size=[.04, 0, 0], fromto=[x0, y, h, x1, y, h], rgba=white, **visual())
        wb.add_geom(type=CAP, size=[.04, 0, 0], fromto=[x0, y, 0, x1, y, 0], rgba=white, **visual())
    wb.add_geom(type=CAP, size=[.06, 0, 0], fromto=[x0, -hw, h, x0, hw, h], rgba=white)
    wb.add_geom(type=CAP, size=[.04, 0, 0], fromto=[x1, -hw, h, x1, hw, h], rgba=white, **visual())
    # Paredes de red que SI frenan la pelota (casi transparentes y blandas)
    soft = dict(rgba=[1, 1, 1, .10], solref=[.05, 2.0], friction=[1.5, .02, .005])
    wb.add_geom(type=BOX, pos=[x1, 0, h/2], size=[.02, hw, h/2], **soft)
    wb.add_geom(type=BOX, pos=[(x0+x1)/2, 0, h], size=[GOAL_D/2, hw, .02], **soft)
    for y in (-hw, hw):
        wb.add_geom(type=BOX, pos=[(x0+x1)/2, y, h/2], size=[GOAL_D/2, .02, h/2], **soft)
    # Hilos de la red (solo visuales)
    net = dict(size=[.007, 0, 0], rgba=[.9, .9, .9, 1])
    for y in np.arange(-hw, hw + .01, .25):            # fondo, vertical
        wb.add_geom(type=CAP, fromto=[x1, y, 0, x1, y, h], **net, **visual())
    for z in np.arange(0, h + .01, .25):               # fondo, horizontal
        wb.add_geom(type=CAP, fromto=[x1, -hw, z, x1, hw, z], **net, **visual())
    for x in np.arange(x0, x1 + .01, .25):             # techo
        wb.add_geom(type=CAP, fromto=[x, -hw, h, x, hw, h], **net, **visual())
    for y in np.arange(-hw, hw + .01, .25):
        wb.add_geom(type=CAP, fromto=[x0, y, h, x1, y, h], **net, **visual())
    for y in (-hw, hw):                                # laterales
        for z in np.arange(0, h + .01, .25):
            wb.add_geom(type=CAP, fromto=[x0, y, z, x1, y, z], **net, **visual())
        for x in np.arange(x0, x1 + .01, .25):
            wb.add_geom(type=CAP, fromto=[x, y, 0, x, y, h], **net, **visual())

def build():
    world = mujoco.MjSpec()
    world.option.timestep = 0.002
    world.stat.extent = 3
    world.visual.global_.offwidth, world.visual.global_.offheight = W, H
    world.visual.headlight.ambient = [.45, .45, .45]
    world.visual.headlight.diffuse = [.7, .7, .7]
    world.visual.rgba.haze = [.75, .85, .95, 1]

    # Cielo (skybox degradado) y pasto (grilla de ajedrez verde)
    world.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=[.25, .5, .85], rgb2=[.85, .93, 1], width=512, height=3072)
    world.add_texture(name="grass", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                      rgb1=[.16, .50, .16], rgb2=[.22, .60, .22], width=512, height=512)
    mat = world.add_material(name="grass", texuniform=True, texrepeat=[1, 1], reflectance=0.0)
    mat.textures = ["", "grass", "", "", "", "", "", "", "", ""]
    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                             pos=[3, 0, 0], size=[40, 40, .05], material="grass")

    # Luz del sol
    world.worldbody.add_light(pos=[2.5, 0, 8], dir=[0, 0, -1],
                              type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=0)
    # Linea de gol y punto penal (solo visuales)
    world.worldbody.add_geom(type=BOX, pos=[GOAL_X, 0, .003], size=[.04, 6, .002],
                             rgba=[1, 1, 1, 1], **visual())
    world.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[BALL_POS[0], BALL_POS[1], .003],
                             size=[.12, .002, 0], rgba=[1, 1, 1, 1], **visual())

    def add_robot(prefix, x, yaw):
        child = mm.get("unitree_h1").spec("h1")           # solo el robot, sin escena
        for j in list(child.body("pelvis").joints):       # sacar freejoint -> base fija
            child.delete(j)
        frame = world.worldbody.add_frame(pos=[x, 0, 0], euler=[0, 0, yaw])
        world.attach(child, frame=frame, prefix=prefix)

    add_robot("kicker_", -1.0, KICK_YAW)                   # mira a +x
    add_robot("keeper_", GOAL_X - 0.7, 180.0)              # mira a -x

    # Pelota: solref blando + poco amortiguado => rebota
    ball = world.worldbody.add_body(name="ball", pos=[BALL_POS[0], BALL_POS[1], 0.11])
    ball.add_freejoint()
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.11, 0, 0],
                  mass=0.43, rgba=[1, 1, 1, 1], friction=[0.8, 0.005, 0.0001],
                  solref=[0.004, 0.25], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    add_goal(world)
    return world.compile()

def pd(m, d, name, q_des, kp, kd):
    jid, aid = m.joint(name).id, m.actuator(name).id
    qadr, vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    tau = kp * (q_des - d.qpos[qadr]) - kd * d.qvel[vadr]
    d.ctrl[aid] = np.clip(tau, *m.actuator_ctrlrange[aid])

def control(m, d):
    t = d.time
    k, g = "kicker_", "keeper_"
    # --- Pateador: backswing (0.3-0.9s) y latigazo (0.9-1.3s)
    if t < 0.3:    hip, knee = 0.0, 0.0
    elif t < 0.9:  hip, knee = 0.5, 1.2
    elif t < 1.3:  hip, knee = -1.0, STRIKE_KNEE
    else:          hip, knee = 0.0, 0.2
    pd(m, d, k+"right_hip_pitch", hip, 1500, 25)
    pd(m, d, k+"right_knee", knee, 1500, 25)
    pd(m, d, k+"right_ankle", 0.0, 60, 3)
    for n in ("left_hip_pitch", "left_knee", "left_ankle", "left_hip_roll",
              "left_hip_yaw", "right_hip_roll", "right_hip_yaw", "torso"):
        pd(m, d, k+n, 0.0, 300, 10)
    # --- Arquero: brazos abiertos, resto quieto
    pd(m, d, g+"left_shoulder_roll", 1.5, 60, 4)
    pd(m, d, g+"right_shoulder_roll", -1.5, 60, 4)
    for n in ("left_shoulder_pitch", "right_shoulder_pitch", "left_elbow", "right_elbow"):
        pd(m, d, g+n, 0.0, 40, 3)
    for n in ("left_hip_pitch", "left_knee", "left_ankle", "right_hip_pitch", "right_knee",
              "right_ankle", "left_hip_roll", "right_hip_roll", "left_hip_yaw",
              "right_hip_yaw", "torso"):
        pd(m, d, g+n, 0.0, 300, 10)

def new_state():
    return {"gol_t": None}

def step(m, d, st):
    control(m, d)
    mujoco.mj_step(m, d)
    p = d.body("ball").xpos
    # Gol: la pelota cruza la linea entre los postes y por debajo del travesano
    if st["gol_t"] is None and p[0] > GOAL_X + .11 and abs(p[1]) < GOAL_HW and p[2] < GOAL_H:
        st["gol_t"] = d.time

def set_camera(cam, distance=11.0):
    """Toma lateral-diagonal: pateador a la izquierda, arco a la derecha, todo en cuadro."""
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [3.0, 0.0, 0.9]
    cam.distance = distance
    cam.azimuth = 70
    cam.elevation = -12

def draw_gol(img, t_since):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    size = int(150 + 12 * np.sin(t_since * 12))          # latido
    font = None
    for path in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try: font = ImageFont.truetype(path, size); break
        except OSError: pass
    if font is None: font = ImageFont.load_default(size=size)
    dr.text((W // 2, int(H * .28)), "GOOOOL!", font=font, anchor="mm",
            fill=(255, 214, 0), stroke_width=8, stroke_fill=(20, 20, 20))
    return np.asarray(im)

def make_video(path=VIDEO):
    import imageio
    m = build(); d = mujoco.MjData(m); st = new_state()
    cam = mujoco.MjvCamera(); set_camera(cam, 11.0)
    r = mujoco.Renderer(m, H, W)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as w:
        for i in range(int(SIM_TIME * FPS)):
            while d.time < i / FPS:
                step(m, d, st)
            r.update_scene(d, cam)
            img = r.render()
            if st["gol_t"] is not None:
                img = draw_gol(img, d.time - st["gol_t"])
            w.append_data(img)
    print(f"Video guardado en {path} | gol: {st['gol_t'] is not None}")

def run_viewer():
    import mujoco.viewer
    m = build(); d = mujoco.MjData(m); st = new_state()
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        set_camera(v.cam, 13.0)
        while v.is_running():
            t0 = time.time()
            step(m, d, st); v.sync()
            if d.time > SIM_TIME:                           # repetir el penal
                mujoco.mj_resetData(m, d); st["gol_t"] = None
            time.sleep(max(0, m.opt.timestep - (time.time() - t0)))

if __name__ == "__main__":
    if "--no-video" not in sys.argv:
        try:
            make_video()
        except Exception as e:
            print("No se pudo generar el video:", repr(e))
            print("Proba: pip install imageio imageio-ffmpeg pillow  (o MUJOCO_GL=egl python h1_penal.py)")
    if "--no-viewer" not in sys.argv:
        run_viewer()