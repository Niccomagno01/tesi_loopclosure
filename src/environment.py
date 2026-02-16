# Classe che contiene gli ostacoli (gestiti con Shapely), i confini e funzioni di utilità geometriche minime

import matplotlib.pyplot as plt  # per disegnare bounds e ostacoli
from shapely.geometry import box  # Primitive geometriche di Shapely necessarie
from shapely.ops import unary_union  # Operazione per unire più geometrie
from shapely.geometry.base import BaseGeometry  # Tipo base per geometrie Shapely
from shapely.geometry import Point, Polygon, LineString  # nuove primitive per forme non rettangolari
from typing import List, Optional  # Tipi per annotazioni statiche
from shapely.errors import TopologicalError  # eccezione specifica per operazioni geometriche


class Environment:
    def __init__(self):
        # Inizializza la lista di ostacoli presenti nell'ambiente
        self.obstacles: List[BaseGeometry] = []  # Lista di ostacoli (oggetti Shapely)
        # Inizializza i confini dell'ambiente (rettangolo), assente all'inizio
        self.bounds: Optional[BaseGeometry] = None  # Confini dell’ambiente (oggetto Shapely)
        # Cache per l'unary_union degli ostacoli (ricostruita on-demand)
        self._union_cache: Optional[BaseGeometry] = None
        self._union_dirty: bool = True

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Imposta i confini dell’ambiente come un rettangolo axis-aligned."""
        self.bounds = box(float(xmin), float(ymin), float(xmax), float(ymax))
        # Bounds non influisce sulla union degli ostacoli

    def add_rectangle(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Aggiunge un ostacolo rettangolare all’ambiente."""
        self.obstacles.append(box(float(xmin), float(ymin), float(xmax), float(ymax)))
        self._union_dirty = True

    # --- Nuove forme di ostacolo ---
    def add_circle(self, cx: float, cy: float, radius: float, *, resolution: int = 32) -> None:
        """Aggiunge un ostacolo circolare (approssimato come poligono) centrato in (cx,cy)."""
        r = max(1e-6, float(radius))
        self.obstacles.append(Point(float(cx), float(cy)).buffer(r, resolution=resolution))
        self._union_dirty = True

    def add_polygon(self, vertices: List[tuple]) -> None:
        """Aggiunge un ostacolo poligonale generico (lista di vertici (x,y))."""
        if not vertices:
            return
        self.obstacles.append(Polygon([(float(x), float(y)) for x, y in vertices]))
        self._union_dirty = True

    def add_wall(self, x0: float, y0: float, x1: float, y1: float, thickness: float = 0.10) -> None:
        """Aggiunge un muro sottile tra (x0,y0)-(x1,y1) bufferizzato con spessore indicato."""
        t = max(1e-6, float(thickness))
        seg = LineString([(float(x0), float(y0)), (float(x1), float(y1))])
        self.obstacles.append(seg.buffer(0.5 * t, cap_style='square', join_style='mitre'))
        self._union_dirty = True

    def obstacles_union(self) -> Optional[BaseGeometry]:
        """Unisce tutti gli ostacoli in un'unica geometria per intersezioni più veloci; None se non ci sono ostacoli.
        La union è cache-ata e ricostruita solo quando la lista degli ostacoli cambia."""
        if not self.obstacles:
            self._union_cache = None
            self._union_dirty = False
            return None
        if self._union_cache is None or self._union_dirty:
            self._union_cache = unary_union(self.obstacles)
            self._union_dirty = False
        return self._union_cache

    def first_intersection_with_line(self, line: LineString):
        """
        Restituisce il punto di intersezione più vicino all'origine del LineString (line.coords[0]),
        come (x, y) tuple, oppure None se non ci sono intersezioni.
        """
        union = self.obstacles_union()
        if union is None:
            return None

        inter = line.intersection(union)
        if inter.is_empty:
            return None

        # Possibili tipi: Point, MultiPoint, LineString, MultiLineString, GeometryCollection
        origin = Point(line.coords[0])

        try:
            # Caso semplice: un solo punto
            if isinstance(inter, Point):
                return float(inter.x), float(inter.y)
            # Più punti: scegli il più vicino all'origine
            if getattr(inter, 'geom_type', '') == 'MultiPoint':
                best = None
                best_d = float('inf')
                for pt in inter.geoms:  # type: ignore[attr-defined]
                    d = origin.distance(pt)
                    if d < best_d:
                        best_d = d
                        best = pt
                if best is not None:
                    return float(best.x), float(best.y)
            # Segmenti/collezioni: prendi il punto su 'inter' più vicino all'origine del raggio
            from shapely.ops import nearest_points  # import locale per evitare dipendenza forte a livello modulo
            _, p = nearest_points(origin, inter)
            return float(p.x), float(p.y)
        except (TopologicalError, AttributeError, TypeError, ValueError):
            # Fallback robusto: proietta le coordinate del tipo di geometria in una lista e scegli la più vicina
            def _iter_points(g):
                gt = getattr(g, 'geom_type', '')
                if gt == 'Point':
                    yield g
                elif gt == 'MultiPoint':
                    for h in g.geoms:
                        yield h
                elif gt in ('LineString', 'LinearRing'):
                    for (x, y) in g.coords:
                        yield Point(x, y)
                elif gt == 'MultiLineString':
                    for h in g.geoms:
                        yield from _iter_points(h)
                elif gt == 'GeometryCollection':
                    for h in g.geoms:
                        yield from _iter_points(h)
            best = None
            best_d = float('inf')
            for pt in _iter_points(inter):
                d = origin.distance(pt)
                if d < best_d:
                    best_d = d
                    best = pt
            if best is None:
                return None
            return float(best.x), float(best.y)

    def plot(self, ax=None, facecolor: str = 'lightgrey', edgecolor: str = 'k') -> None:
        """Disegna bounds e ostacoli (se ax è None crea una figura nuova)."""
        own_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            own_fig = True
        # Disegna bounds (se presenti)
        if self.bounds is not None:
            x, y = self.bounds.exterior.xy  # type: ignore[attr-defined]
            ax.plot(x, y, color=edgecolor, linewidth=1.0, zorder=0)
            ax.fill(x, y, alpha=0.04, facecolor=facecolor, edgecolor='none', zorder=0)
        # Disegna ogni ostacolo come poligono riempito
        for poly in self.obstacles:
            x, y = poly.exterior.xy  # type: ignore[attr-defined]
            ax.fill(x, y, alpha=0.6, facecolor='tab:gray', edgecolor=edgecolor, linewidth=1.0, zorder=1)
        if own_fig:
            ax.set_aspect('equal', 'box')
            plt.show()
