# render.py
import pygame
from Box2D.b2 import polygonShape, circleShape

PPM = 50.0
ORIGIN = (100, 500)
def draw_world(screen, world, width, height):
    screen.fill((30,30,30))
    
    for body in world.bodies:
        for fix in body.fixtures:
            shape = fix.shape
            if isinstance(shape, polygonShape):
                # 월드 좌표 → 화면 좌표 변환 함수
                def W2S(v):
                    x, y = body.transform * v
                    sx = ORIGIN[0] + x * PPM
                    sy = ORIGIN[1] - y * PPM
                    return (int(sx), int(sy))
                
                verts = [W2S(v) for v in shape.vertices]
                pygame.draw.polygon(screen, (200,200,200), verts)
            elif isinstance(shape, circleShape):
                # 월드 동일하게 변환
                x, y = body.transform * shape.pos
                sx = ORIGIN[0] + x * PPM
                sy = ORIGIN[1] - y * PPM
                pygame.draw.circle(
                    screen, (200,200,200),
                    (int(sx), int(sy)),
                    int(shape.radius * PPM))