import numpy as np
import pygame
import taichi as ti
import glm


ti.init(ti.cuda)
pygame.init()
pygame.font.init()

VIEWPORT_WIDTH, VIEWPORT_HEIGHT = 1000, 1000
WIDTH, HEIGHT = 1000, 1000
running = True

buffer = np.zeros((VIEWPORT_WIDTH, VIEWPORT_HEIGHT, 3), dtype=np.uint8)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()

pgVec2 = pygame.math.Vector2
pgVec3 = pygame.math.Vector3


camera_values = ti.types.struct(
    vettical_fov=ti.float64,
    near_clip=ti.float64,
    far_clip=ti.float64,

    projection=ti.math.mat4,
    view=ti.math.mat4,
    inverse_projection=ti.math.mat4,
    inverse_view=ti.math.mat4,

    position=ti.math.vec3,
    direction=ti.math.vec3,
)

Ray = ti.types.struct(
    origin=ti.math.vec3,
    direction=ti.math.vec3
)

Sphere = ti.types.struct(
    position=ti.math.vec3,
    albedo=ti.math.vec3,
    radius=ti.types.float64
)



font = pygame.font.SysFont("consolas", 24)


@ti.func
def trace_ray(ray: Ray):
    radius = 0.5

    a = ray.direction.dot(ray.direction)
    b = 2 * ray.origin.dot(ray.direction)
    c = ray.origin.dot(ray.origin) - radius * radius

    discriminant = b * b - 4.0 * a * c
    c_mult = ti.math.vec3([1, 1, 1])
    if discriminant < 0:
        c_mult = ti.math.vec3([0, 0, 0])

    t0 = (-b + ti.sqrt(discriminant)) / (2 * a)
    closestT = (-b - ti.sqrt(discriminant)) / (2 * a)

    hitPoint = ray.origin + ray.direction * closestT
    normal = hitPoint.normalized()

    lightDir = ti.math.vec3([-1, -1, -1]).normalized()

    d = ti.max(normal.dot(-lightDir), 0)

    sphereColor = ti.math.vec3([1, 0, 1])
    sphereColor *= d
    return sphereColor * c_mult


@ti.kernel
def render_fast_gpu(buff: ti.types.ndarray(), camera: camera_values):
    ray_origin = camera.position

    ray = Ray(origin=ray_origin)

    for y in range(VIEWPORT_HEIGHT):
        for x in range(VIEWPORT_WIDTH):
            coord = ti.math.vec2(x / VIEWPORT_WIDTH, y / VIEWPORT_HEIGHT)
            coord = coord * 2 - 1

            # calculate ray direction here because python is too slow for that
            target = camera.inverse_projection @ ti.Vector([coord.x, coord.y, 1, 1])
            ray_direction = camera.inverse_view @ ti.Vector(
                [target[0] / target[3], target[1] / target[3], target[2] / target[3], 0])

            ray.direction = ray_direction.xyz

            c = trace_ray(ray)
            buff[x, y, 0] = ti.max(ti.min(c[0] * 255, 255), 0)
            buff[x, y, 1] = ti.max(ti.min(c[1] * 255, 255), 0)
            buff[x, y, 2] = ti.max(ti.min(c[2] * 255, 255), 0)


class Renderer:
    def __init__(self, buff):
        self.final_image = buff

    def render(self, cam):
        taichi_array = ti.ndarray(dtype=ti.uint8, shape=(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, 3))
        taichi_array.from_numpy(self.final_image)
        render_fast_gpu(taichi_array, cam.camera_struct)
        self.final_image = taichi_array.to_numpy()
        return self.final_image


class Camera:
    def __init__(self, vertical_fov, near_clip, far_clip):
        self.camera_struct = camera_values()

        self.camera_struct.vertical_fov = vertical_fov
        self.camera_struct.near_clip = near_clip
        self.camera_struct.far_clip = far_clip

        self.camera_struct.direction = ti.math.vec3(0, 0, -1)
        self.camera_struct.position = ti.math.vec3(0, 0, 3)

        self.last_mouse_pos = ti.math.vec2(0, 0)
        self.last_visible_cursor_pos: ti.math.vec2 = None

        self.recalculate_projection()
        self.recalculate_view()

    def on_update(self, ts):
        mouse_pos = ti.math.vec2(pygame.mouse.get_pos())
        delta = ti.math.vec2(mouse_pos - self.last_mouse_pos) * 0.002
        self.last_mouse_pos = mouse_pos

        if not pygame.mouse.get_pressed(3)[2]:
            if self.last_visible_cursor_pos:
                pygame.mouse.set_pos((self.last_visible_cursor_pos.x, self.last_visible_cursor_pos.y))
                self.last_visible_cursor_pos = None
            pygame.mouse.set_visible(True)
            return
        if not self.last_visible_cursor_pos:
            self.last_visible_cursor_pos = mouse_pos
        pygame.mouse.set_visible(False)

        moved = False

        up_direction = ti.math.vec3(0, 1, 0)
        right_direction = self.camera_struct.direction.cross(up_direction)

        speed = 5

        if pygame.key.get_pressed()[pygame.K_w]:
            self.camera_struct.position += self.camera_struct.direction * speed * ts
            moved = True
        elif pygame.key.get_pressed()[pygame.K_s]:
            self.camera_struct.position -= self.camera_struct.direction * speed * ts
            moved = True

        if pygame.key.get_pressed()[pygame.K_a]:
            self.camera_struct.position -= right_direction * speed * ts
            moved = True
        elif pygame.key.get_pressed()[pygame.K_d]:
            self.camera_struct.position += right_direction * speed * ts
            moved = True

        if pygame.key.get_pressed()[pygame.K_q]:
            self.camera_struct.position -= up_direction * speed * ts
            moved = True
        elif pygame.key.get_pressed()[pygame.K_e]:
            self.camera_struct.position += up_direction * speed * ts
            moved = True

        if delta.x != 0 or delta.y != 0:
            pitch_delta = delta.y * self.get_rotation_speed()  # Example value, replace with your calculation
            yaw_delta = delta.x * self.get_rotation_speed()  # Example value, replace with your calculation
            rightDirection_glm = glm.vec3(right_direction.to_list())
            direction_glm = glm.vec3(self.camera_struct.direction.to_list())

            q = glm.normalize(glm.cross(glm.angleAxis(-pitch_delta, rightDirection_glm),
                                        glm.angleAxis(yaw_delta, glm.vec3(0.0, 1.0, 0.0))))

            # Convert the direction vector to a quaternion
            direction_quat = glm.quat(0.0, direction_glm)

            # Rotate the direction quaternion using the calculated quaternion
            rotated_direction_quat = q * direction_quat * glm.conjugate(q)

            # Convert the rotated quaternion back to a vector
            new_direction_glm = glm.vec3(rotated_direction_quat.x, rotated_direction_quat.y, rotated_direction_quat.z)
            self.camera_struct.direction = ti.math.vec3(new_direction_glm.to_list())
            moved = True

        if moved:
            self.recalculate_view()

    def get_rotation_speed(self):
        return 0.3

    def recalculate_view(self):
        position = glm.vec3(self.camera_struct.position.to_list())
        forward_direction = glm.vec3(self.camera_struct.direction.to_list())
        view = glm.lookAt(position, position + forward_direction, glm.vec3(0, 1, 0))
        self.camera_struct.view = ti.math.mat4(view.to_list())
        self.camera_struct.inverse_view = ti.math.mat4(glm.inverse(view).to_list())

    def recalculate_projection(self):
        v_fov = float(self.camera_struct.vertical_fov)
        near_clip = float(self.camera_struct.near_clip)
        far_clip = float(self.camera_struct.far_clip)
        projection = glm.perspectiveFov(glm.radians(v_fov), VIEWPORT_WIDTH, VIEWPORT_HEIGHT, near_clip, far_clip)

        self.camera_struct.projection = ti.math.mat4(projection.to_list())
        self.camera_struct.inverse_projection = ti.math.mat4(glm.inverse(projection).to_list())


renderer = Renderer(buffer)
camera = Camera(45, 0.1, 100)

getTicksLastFrame = 0
while running:
    screen.fill(0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    rendered_image = renderer.render(camera)
    screen.blit(pygame.transform.scale(pygame.pixelcopy.make_surface(rendered_image), (WIDTH, HEIGHT)), (0, 0))

    p = camera.camera_struct.position
    d = camera.camera_struct.direction

    a = font.render(f"{round(p.x, 2)}x {round(p.y, 2)}y {round(p.z, 2)}z", True, (255, 255, 255))
    b = font.render(f"{round(d.x, 2)}pitch {round(d.y, 2)}yaw", True, (255, 255, 255))
    screen.blit(a, (0, 0))
    screen.blit(b, (0, 20))

    pygame.display.flip()
    t = pygame.time.get_ticks()
    dt = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t
    camera.on_update(dt)
    clock.tick(0)
    fps = clock.get_fps()
    if fps != 0:
        pygame.display.set_caption(f"{1000 / fps}ms")
