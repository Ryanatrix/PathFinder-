import pygame as pg
import os
import math

ASSETS = {
    "drone_img": "assets/drone.png",
    "obstacle_img": "assets/block.png",
    "background_img": "assets/mars.jpg",
    "smoke_img": "assets/blackSmoke00.png",
    "light_mask": "assets/light_350_soft.png",
}

def _find_asset_file(base_dir, asset_path):
    if not asset_path:
        return None
    if os.path.isabs(asset_path) and os.path.exists(asset_path):
        return asset_path
    c = os.path.join(base_dir, asset_path)
    if os.path.exists(c):
        return c
    bn = os.path.basename(asset_path)
    c = os.path.join(base_dir, "assets", bn)
    if os.path.exists(c):
        return c
    stem, _ = os.path.splitext(bn)
    ad = os.path.join(base_dir, "assets")
    if os.path.isdir(ad):
        for f in os.listdir(ad):
            if stem.lower() in f.lower():
                p = os.path.join(ad, f)
                if os.path.isfile(p):
                    return p
    return None


class Drone(pg.sprite.Sprite):
    def __init__(self, viewer, rover):
        self.groups = viewer.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.viewer = viewer
        self.rover = rover
        self.image = viewer.drone_img
        self.rect = self.image.get_rect()

    def update(self):
        sx, sy = self.viewer.world_to_screen(self.rover.pos[0], self.rover.pos[1])
        self.rect.center = (sx, sy)


class Base(pg.sprite.Sprite):
    def __init__(self, viewer, rover):
        self.groups = viewer.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.viewer = viewer
        self.rover = rover
        self.image = viewer.base_img
        self.rect = self.image.get_rect()

    def update(self):
        sx, sy = self.viewer.world_to_screen(self.rover.base[0], self.rover.base[1])
        self.rect.center = (sx, sy)


class Obstacle(pg.sprite.Sprite):
    def __init__(self, viewer, x, y):
        self.groups = viewer.all_sprites, viewer.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.viewer = viewer
        self.image = viewer.obstacle_img
        self.rect = self.image.get_rect()
        sx, sy = viewer.world_to_screen(x, y)
        self.rect.center = (sx, sy)


class Visualizer:
    def __init__(self, cfg, grid, rover):
        pg.init()
        self.cfg = cfg
        self.grid = grid
        self.rover = rover

        self.world_w = cfg.map_w
        self.world_h = cfg.map_h
        self.sidebar_w = cfg.sidebar_width

        self.screen_w = self.world_w + self.sidebar_w
        self.screen_h = self.world_h

        self.screen = pg.display.set_mode((self.screen_w, self.screen_h), pg.RESIZABLE)

        self.render_surface = pg.Surface((self.world_w, self.world_h))
        self.clock = pg.time.Clock()

        self.camera_scale = 1.0
        self.camera_offset = [0.0, 0.0]

        self.sprite_scale = float(cfg.sprite_scale)

        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.drone_img_orig = self._load(_find_asset_file(base_dir, ASSETS["drone_img"]))
        self.obstacle_img_orig = self._load(_find_asset_file(base_dir, ASSETS["obstacle_img"]))
        self.bck_img_orig = self._load(_find_asset_file(base_dir, ASSETS["background_img"]),
                                       size=(self.world_w, self.world_h), alpha=False)
        self.light_mask_img = self._load(_find_asset_file(base_dir, ASSETS["light_mask"]),
                                         size=(200, 200))

        base_raw = pg.Surface((8, 8), pg.SRCALPHA)
        pg.draw.circle(base_raw, (255, 255, 255), (4, 4), 4)
        self.base_img_orig = base_raw

        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()

        self._prepare_scaled_assets()

        self.base_marker = Base(self, rover)
        self.player = Drone(self, rover)

        self.world_fog = pg.Surface((self.world_w, self.world_h), pg.SRCALPHA)
        self.world_fog.fill((0, 0, 0, 255))

        self.font = pg.font.SysFont("Consolas", 18)

        self.center_on_rover()

    def _load(self, path, size=None, alpha=True):
        try:
            img = pg.image.load(path)
            img = img.convert_alpha() if alpha else img.convert()
            if size:
                img = pg.transform.smoothscale(img, size)
            return img
        except:
            return pg.Surface((32, 32), pg.SRCALPHA)

    def _prepare_scaled_assets(self):
        s = self.camera_scale

        self.bck_img = pg.transform.smoothscale(
            self.bck_img_orig, (int(self.world_w * s), int(self.world_h * s))
        )

        self.drone_img = pg.transform.smoothscale(self.drone_img_orig, (12, 12))
        self.obstacle_img = pg.transform.smoothscale(self.obstacle_img_orig, (12, 12))
        self.base_img = pg.transform.smoothscale(self.base_img_orig, (8, 8))

    def world_to_screen(self, wx, wy):
        s = self.camera_scale
        return int(wx * s + self.camera_offset[0]), int(wy * s + self.camera_offset[1])

    def screen_to_world(self, sx, sy):
        s = self.camera_scale
        return (sx - self.camera_offset[0]) / s, (sy - self.camera_offset[1]) / s

    def center_on_rover(self):
        s = self.camera_scale
        rx = self.rover.pos[0] * s
        ry = self.rover.pos[1] * s
        self.camera_offset[0] = self.world_w / 2 - rx
        self.camera_offset[1] = self.world_h / 2 - ry

    def zoom_at(self, pos, factor):
        new_scale = self.camera_scale * factor
        new_scale = max(0.25, min(6.0, new_scale))
        if abs(new_scale - self.camera_scale) < 1e-6:
            return
        sx, sy = pos
        wx, wy = self.screen_to_world(sx, sy)

        self.camera_scale = new_scale
        self._prepare_scaled_assets()

        s = self.camera_scale
        self.camera_offset[0] = sx - wx * s
        self.camera_offset[1] = sy - wy * s

        for spr in self.all_sprites:
            spr.update()

    def zoom_to_rover(self, factor):
        self.zoom_at((self.world_w // 2, self.world_h // 2), factor)

    def reveal_visible(self):
        if not self.light_mask_img:
            return
        m = self.light_mask_img
        for cx, cy in self.rover.revealed_centers:
            self.world_fog.blit(m, (cx - 100, cy - 100), special_flags=pg.BLEND_RGBA_SUB)

    def draw(self):
        bx = -int(self.camera_offset[0])
        by = -int(self.camera_offset[1])

        self.render_surface.blit(self.bck_img, (bx, by))

        self.all_sprites.update()
        for spr in self.all_sprites:
            self.render_surface.blit(spr.image, spr.rect)

        for a, b, col in self.rover.trace:
            ax, ay = self.world_to_screen(a[0], a[1])
            bx2, by2 = self.world_to_screen(b[0], b[1])
            pg.draw.line(self.render_surface, col, (ax, ay), (bx2, by2), 2)

        self.reveal_visible()
        fog_scaled = pg.transform.smoothscale(
            self.world_fog,
            (int(self.world_w * self.camera_scale), int(self.world_h * self.camera_scale))
        )
        self.render_surface.blit(fog_scaled, (bx, by))

        self.screen.blit(self.render_surface, (0, 0))

        pg.draw.rect(self.screen, (30, 30, 30), (self.world_w, 0, self.sidebar_w, self.screen_h))

        pg.display.flip()
        self.clock.tick(self.cfg.fps)