import taichi as ti
import math

ti.init(arch=ti.cpu)

base_cube = ti.Vector.field(3, ti.f32, shape=8)
scr = ti.Vector.field(2, ti.f32, shape=8)

edges = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c, s = ti.cos(rad), ti.sin(rad)
    return ti.Matrix([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([[1,0,0,-eye_pos[0]],[0,1,0,-eye_pos[1]],[0,0,1,-eye_pos[2]],[0,0,0,1]])

@ti.func
def get_projection_matrix(fov, ar, n, f):
    n = -n
    f = -f
    rad = fov*math.pi/180
    t = ti.tan(rad/2)*abs(n)
    b,l,r = -t,-ar*t,ar*t
    M_p2o = ti.Matrix([[n,0,0,0],[0,n,0,0],[0,0,n+f,-n*f],[0,0,1,0]])
    scl = ti.Matrix([[2/(r-l),0,0,0],[0,2/(t-b),0,0],[0,0,2/(n-f),0],[0,0,0,1]])
    tra = ti.Matrix([[1,0,0,-(r+l)/2],[0,1,0,-(t+b)/2],[0,0,1,-(n+f)/2],[0,0,0,1]])
    return scl@tra@M_p2o

# 拆分：单独计算屏幕坐标，不再被其他kernel调用
@ti.kernel
def calc_screen(in_vert:ti.template(), ang:ti.f32):
    eye = ti.Vector([0,0,5])
    mvp = get_projection_matrix(45,1,0.1,50) @ get_view_matrix(eye) @ get_model_matrix(ang)
    for i in range(8):
        p = ti.Vector([in_vert[i][0],in_vert[i][1],in_vert[i][2],1])
        clip = mvp@p
        ndc = clip/clip[3]
        scr[i] = [(ndc[0]+1)/2, (ndc[1]+1)/2]

# 插值放在python层，不再写在内核里
def main():
    cube_p = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
              [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
    for i in range(8):
        base_cube[i] = cube_p[i]

    ang0 = 0.0   # R0初始姿态 蓝色
    ang1 = 90.0  # R1终止姿态 红色
    t = 0.0
    dt = 0.006

    gui = ti.GUI("Cube Lerp Demo", res=(700,700))
    while gui.running:
        # 1.绘制起点蓝立方体 R0
        calc_screen(base_cube, ang0)
        for a,b in edges:
            gui.line(scr[a],scr[b],radius=2,color=0x44aaff)

        # 2.绘制终点红立方体 R1
        calc_screen(base_cube, ang1)
        for a,b in edges:
            gui.line(scr[a],scr[b],radius=2,color=0xff4444)

        # 3.Python层插值角度，中间黄立方体动态过渡
        cur_ang = ang0*(1-t)+ang1*t
        calc_screen(base_cube, cur_ang)
        for a,b in edges:
            gui.line(scr[a],scr[b],radius=2,color=0xffdd77)

        t += dt
        if t>1.0:
            t = 0.0
        gui.show()

if __name__ == '__main__':
    main()