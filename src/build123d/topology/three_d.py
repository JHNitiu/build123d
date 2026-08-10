"""
build123d topology

name: three_d.py
by:   Gumyr
date: January 07, 2025

desc:

This module defines the `Solid` class and associated methods for creating, manipulating, and
querying three-dimensional solid geometries in the build123d CAD system. It provides powerful tools
for constructing complex 3D models, including operations such as extrusion, sweeping, filleting,
chamfering, and Boolean operations. The module integrates with OpenCascade to leverage its robust
geometric kernel for precise 3D modeling.

Key Features:
- **Solid Class**:
  - Represents closed, bounded 3D shapes with methods for volume calculation, bounding box
    computation, and validity checks.
  - Includes constructors for primitive solids (e.g., box, cylinder, cone, torus) and advanced
    operations like lofting, revolving, and sweeping profiles along paths.

- **Mixin3D**:
  - Adds shared methods for operations like filleting, chamfering, splitting, and hollowing solids.
  - Supports advanced workflows such as finding maximum fillet radii and extruding with rotation or
    taper.

- **Boolean Operations**:
  - Provides utilities for union, subtraction, and intersection of solids.

- **Thickening and Offsetting**:
  - Allows transformation of faces or shells into solids through thickening.

This module is essential for generating and manipulating complex 3D geometries in the build123d
library, offering a comprehensive API for CAD modeling.

license:

    Copyright 2025 Gumyr

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from math import copysign, cos, radians, tan
from typing import TYPE_CHECKING, List, Literal, Tuple, cast

import OCP.TopAbs as ta
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepFeat import BRepFeat_MakeDPrism
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp_Face
from OCP.BRepOffset import BRepOffset_MakeOffset, BRepOffset_Skin
from OCP.BRepOffsetAPI import (
    BRepOffsetAPI_DraftAngle,
    BRepOffsetAPI_MakePipeShell,
    BRepOffsetAPI_MakeThickSolid,
)
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeRevol,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus,
    BRepPrimAPI_MakeWedge,
)
from OCP.GeomAbs import GeomAbs_Intersection, GeomAbs_JoinType
from OCP.gp import gp_Ax2, gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.LocOpe import LocOpe_DPrism
from OCP.ShapeFix import ShapeFix_Solid
from OCP.Standard import Standard_Failure, Standard_TypeMismatch
from OCP.StdFail import StdFail_NotDone
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import (
    TopoDS,
    TopoDS_Compound,
    TopoDS_Face,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Wire,
)
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListOfShape
from typing_extensions import Self

from build123d.build_enums import CenterOf, GeomType, Keep, Kind, Transition, Until
from build123d.geometry import (
    DEG2RAD,
    TOLERANCE,
    Axis,
    BoundBox,
    Color,
    Location,
    Matrix,
    OrientedBoundBox,
    Plane,
    Vector,
    VectorLike,
)

from .one_d import Edge, Mixin1D, Wire, topo_explore_connected_faces
from .shape_core import (
    TOPODS,
    Joint,
    Shape,
    ShapeList,
    _sew_topods_faces,
    downcast,
    get_top_level_topods_shapes,
    shapetype,
    unwrap_topods_compound,
    _make_topods_compound_from_shapes,
)
from .two_d import Face, Mixin2D, Shell, faces_are_tangent, sort_wires_by_build_order
from .utils import (
    _extrude_topods_shape,
    _make_loft,
    find_max_dimension,
)
from .zero_d import Vertex
import networkx as nx
from networkx import bfs_layers
import matplotlib.pyplot as plt
from ocp_vscode import Render, show
from build123d.geometry import Rot, Location, Vector, Pos
import csv
from enum import Enum, auto
import os
import math

if TYPE_CHECKING:  # pragma: no cover
    from .composite import Compound, Part  # pylint: disable=R0801


class Mixin3D(Shape[TOPODS]):
    """Additional methods to add to 3D Shape classes"""

    find_intersection_points = Mixin2D.find_intersection_points

    # ---- Properties ----

    @property
    def _dim(self) -> int | None:
        """Dimension of Solids"""
        return 3

    # ---- Class Methods ----

    @classmethod
    def cast(cls, obj: TopoDS_Shape) -> Self:
        "Returns the right type of wrapper, given a OCCT object"

        # define the shape lookup table for casting
        constructor_lut = {
            ta.TopAbs_VERTEX: Vertex,
            ta.TopAbs_EDGE: Edge,
            ta.TopAbs_WIRE: Wire,
            ta.TopAbs_FACE: Face,
            ta.TopAbs_SHELL: Shell,
            ta.TopAbs_SOLID: Solid,
        }

        shape_type = shapetype(obj)
        # NB downcast is needed to handle TopoDS_Shape types
        return constructor_lut[shape_type](downcast(obj))

    @classmethod
    def extrude(
        cls, obj: Shape, direction: VectorLike
    ) -> Edge | Face | Shell | Solid | Compound:
        """Unused - only here because Mixin1D is a subclass of Shape"""
        return NotImplemented

    @staticmethod
    def _make_3d_result(shape: TopoDS_Shape) -> Solid | Part:
        """Wrap a 3D operation result as topology, not as the source subclass."""
        result = downcast(shape)

        if isinstance(result, TopoDS_Compound):
            result = downcast(unwrap_topods_compound(result, True))

        if isinstance(result, TopoDS_Compound):
            solids = ShapeList(
                Solid(TopoDS.Solid(s)) for s in get_top_level_topods_shapes(result)
            )
            return cast("Part", Shape.make_composite(solids, 3))

        return Solid(TopoDS.Solid(result))

    # ---- Instance Methods ----

    def center(self, center_of: CenterOf = CenterOf.MASS) -> Vector:
        """Return center of object

        Find center of object

        Args:
            center_of (CenterOf, optional): center option. Defaults to CenterOf.MASS.

        Raises:
            ValueError: Center of GEOMETRY is not supported for this object
            NotImplementedError: Unable to calculate center of mass of this object

        Returns:
            Vector: center
        """
        if center_of == CenterOf.GEOMETRY:
            raise ValueError("Center of GEOMETRY is not supported for this object")
        if center_of == CenterOf.MASS:
            properties = GProp_GProps()
            calc_function = Shape.shape_properties_LUT[shapetype(self.wrapped)]
            assert calc_function is not None
            calc_function(self.wrapped, properties)
            middle = Vector(properties.CentreOfMass())
        else:  # center_of == CenterOf.BOUNDING_BOX:
            middle = self.bounding_box().center()
        return middle

    def chamfer(
        self,
        length: float,
        length2: float | None,
        edge_list: Iterable[Edge],
        face: Face | None = None,
    ) -> Solid | Part:
        """Chamfer

        Chamfers the specified edges of this solid.

        Args:
            length (float): length > 0, the length (length) of the chamfer
            length2 (Optional[float]): length2 > 0, optional parameter for asymmetrical
                chamfer. Should be `None` if not required.
            edge_list (Iterable[Edge]): a list of Edge objects, which must belong to
                this solid
            face (Face, optional): identifies the side where length is measured. The edge(s)
                must be part of the face

        Returns:
            Solid | Part:  Chamfered solid or 3D composite
        """
        edge_list = list(edge_list)
        if face:
            if any(edge for edge in edge_list if edge not in face.edges()):
                raise ValueError("Some edges are not part of the face")

        native_edges = [e.wrapped for e in edge_list]

        # make a edge --> faces mapping
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(
            self.wrapped, ta.TopAbs_EDGE, ta.TopAbs_FACE, edge_face_map
        )

        # note: we prefer 'length' word to 'radius' as opposed to FreeCAD's API
        chamfer_builder = BRepFilletAPI_MakeChamfer(self.wrapped)

        if length2:
            distance1 = length
            distance2 = length2
        else:
            distance1 = length
            distance2 = length

        for native_edge in native_edges:
            if face:
                topo_face = face.wrapped
            else:
                topo_face = edge_face_map.FindFromKey(native_edge).First()

            chamfer_builder.Add(
                distance1, distance2, native_edge, TopoDS.Face(topo_face)
            )  # NB: edge_face_map return a generic TopoDS_Shape

        try:
            new_shape = self._make_3d_result(chamfer_builder.Shape())
            if not new_shape.is_valid:
                raise Standard_Failure
        except (StdFail_NotDone, Standard_Failure) as err:
            raise ValueError(
                "Failed creating a chamfer, try a smaller length value(s)"
            ) from err

        return new_shape

    def dprism(
        self,
        basis: Face | None,
        bounds: list[Face | Wire],
        depth: float | None = None,
        taper: float = 0,
        up_to_face: Face | None = None,
        thru_all: bool = True,
        additive: bool = True,
    ) -> Solid:
        """dprism

        Make a prismatic feature (additive or subtractive)

        Args:
            basis (Optional[Face]): face to perform the operation on
            bounds (list[Union[Face,Wire]]): list of profiles
            depth (float, optional): depth of the cut or extrusion. Defaults to None.
            taper (float, optional): in degrees. Defaults to 0.
            up_to_face (Face, optional): a face to extrude until. Defaults to None.
            thru_all (bool, optional): cut thru_all. Defaults to True.
            additive (bool, optional): Defaults to True.

        Returns:
            Solid: prismatic feature
        """
        if isinstance(bounds[0], Wire):
            sorted_profiles = sort_wires_by_build_order(bounds)
            faces = [Face(p[0], p[1:]) for p in sorted_profiles]
        else:
            faces = bounds

        shape: TopoDS_Shape | TopoDS_Solid = self.wrapped
        for face in faces:
            feat = BRepFeat_MakeDPrism(
                shape,
                face.wrapped,
                basis.wrapped if basis else TopoDS_Face(),
                taper * DEG2RAD,
                additive,
                False,
            )

            if up_to_face is not None:
                feat.Perform(up_to_face.wrapped)
            elif thru_all or depth is None:
                feat.PerformThruAll()
            else:
                feat.Perform(depth)

            shape = feat.Shape()

        return self.__class__(shape)

    def fillet(self, radius: float, edge_list: Iterable[Edge]) -> Solid | Part:
        """Fillet

        Fillets the specified edges of this solid.

        Args:
            radius (float): float > 0, the radius of the fillet
            edge_list (Iterable[Edge]): a list of Edge objects, which must belong to this solid

        Returns:
            Solid | Part: Filleted solid or 3D composite
        """
        native_edges = [e.wrapped for e in edge_list]

        fillet_builder = BRepFilletAPI_MakeFillet(self.wrapped)

        for native_edge in native_edges:
            fillet_builder.Add(radius, native_edge)

        try:
            new_shape = self._make_3d_result(fillet_builder.Shape())
            if not new_shape.is_valid:
                raise Standard_Failure
        except (StdFail_NotDone, Standard_Failure) as err:
            raise ValueError(
                f"Failed creating a fillet with radius of {radius}, try a smaller value"
                f" or use max_fillet() to find the largest valid fillet radius"
            ) from err

        return new_shape

    def hollow(
        self,
        faces: Iterable[Face] | None,
        thickness: float,
        tolerance: float = 0.0001,
        kind: Kind = Kind.ARC,
    ) -> Solid:
        """Hollow

        Return the outer shelled solid of self.

        Args:
            faces (Optional[Iterable[Face]]): faces to be removed,
            which must be part of the solid. Can be an empty list.
            thickness (float): shell thickness - positive shells outwards, negative
                shells inwards.
            tolerance (float, optional): modelling tolerance of the method. Defaults to 0.0001.
            kind (Kind, optional): intersection type. Defaults to Kind.ARC.

        Raises:
            ValueError: Kind.TANGENT not supported

        Returns:
            Solid: A hollow solid.
        """
        faces = list(faces) if faces else []
        if kind == Kind.TANGENT:
            raise ValueError("Kind.TANGENT not supported")

        kind_dict = {
            Kind.ARC: GeomAbs_JoinType.GeomAbs_Arc,
            Kind.INTERSECTION: GeomAbs_JoinType.GeomAbs_Intersection,
        }

        occ_faces_list = TopTools_ListOfShape()
        for face in faces:
            occ_faces_list.Append(face.wrapped)

        shell_builder = BRepOffsetAPI_MakeThickSolid()
        shell_builder.MakeThickSolidByJoin(
            self.wrapped,
            occ_faces_list,
            thickness,
            tolerance,
            Intersection=True,
            Join=kind_dict[kind],
        )
        shell_builder.Build()

        if faces:
            return_value = self.__class__.cast(shell_builder.Shape())

        else:  # if no faces provided a watertight solid will be constructed
            shell1 = self.__class__.cast(shell_builder.Shape()).shells()[0].wrapped
            shell2 = self.shells()[0].wrapped  # pylint: disable=no-member

            # s1 can be outer or inner shell depending on the thickness sign
            if thickness > 0:
                sol = BRepBuilderAPI_MakeSolid(shell1, shell2)
            else:
                sol = BRepBuilderAPI_MakeSolid(shell2, shell1)

            # fix needed for the orientations
            return_value = self.__class__.cast(sol.Shape()).fix()

        return return_value

    def _intersect(
        self,
        other: Shape | Vector | Location | Axis | Plane,
        tolerance: float = 1e-6,
        include_touched: bool = False,
    ) -> ShapeList | None:
        """Single-object intersection for Solid.

        Returns same-dimension overlap or crossing geometry:
        - Solid + Solid → Solid (volume overlap)
        - Solid + Face → Face (portion in/on solid)
        - Solid + Edge → Edge (portion through solid)

        Args:
            other: Shape or geometry object to intersect with
            tolerance: tolerance for intersection detection
            include_touched: if True, include boundary contacts
                (shapes touching the solid's surface without penetrating)
        """
        # Convert geometry objects to shapes
        if isinstance(other, Vector):
            other = Vertex(other)
        elif isinstance(other, Location):
            other = Vertex(other.position)
        elif isinstance(other, Axis):
            other = Edge(other)
        elif isinstance(other, Plane):
            other = Face(other)

        def filter_redundant_touches(items: ShapeList) -> ShapeList:
            """Remove vertices/edges that lie on higher-dimensional results."""
            edges = [r for r in items if isinstance(r, Edge)]
            faces = [r for r in items if isinstance(r, Face)]
            solids = [r for r in items if isinstance(r, Solid)]
            return ShapeList(
                r
                for r in items
                if not (
                    isinstance(r, Vertex)
                    and (
                        any(e.distance_to(r) <= tolerance for e in edges)
                        or any(f.distance_to(r) <= tolerance for f in faces)
                        or any(
                            sf.distance_to(r) <= tolerance
                            for s in solids
                            for sf in s.faces()
                        )
                    )
                )
                and not (
                    isinstance(r, Edge)
                    and any(f.distance_to(r.center()) <= tolerance for f in faces)
                )
            )

        results: ShapeList = ShapeList()

        # Trim infinite edges before OCCT operations
        if isinstance(other, Edge) and other.is_infinite:
            bbox = self.bounding_box(optimal=False)
            other = other.trim_infinite(
                bbox.diagonal + (other.center() - bbox.center()).length
            )

        # Solid + Solid/Face/Shell/Edge/Wire: use Common
        if isinstance(other, (Solid, Face, Shell, Edge, Wire)):
            intersection = self._bool_op_list((self,), (other,), BRepAlgoAPI_Common())
            results.extend(intersection.expand())
        # Solid + Vertex: point containment check
        elif isinstance(other, Vertex):
            if self.is_inside(Vector(other), tolerance):
                results.append(other)

        # Delegate to higher-order shapes (Compound)
        # Don't pass include_touched - outer caller handles touches
        else:
            result = other._intersect(self, tolerance, include_touched=False)
            if result:
                results.extend(result)

        # Add boundary contacts if requested (only Solid has touch method)
        if include_touched and isinstance(self, Solid):
            results.extend(self.touch(other, tolerance))
            results = filter_redundant_touches(ShapeList(set(results)))

        return results if results else None

    def is_inside(self, point: VectorLike, tolerance: float = 1.0e-6) -> bool:
        """Returns whether or not the point is inside a solid or compound
        object within the specified tolerance.

        Args:
          point: tuple or Vector representing 3D point to be tested
          tolerance: tolerance for inside determination, default=1.0e-6
          point: VectorLike:
          tolerance: float:  (Default value = 1.0e-6)

        Returns:
          bool indicating whether or not point is within solid

        """
        solid_classifier = BRepClass3d_SolidClassifier(self.wrapped)
        solid_classifier.Perform(gp_Pnt(*Vector(point)), tolerance)

        return solid_classifier.State() == ta.TopAbs_IN or solid_classifier.IsOnAFace()

    def max_fillet(
        self,
        edge_list: Iterable[Edge],
        tolerance=0.1,
        max_iterations: int = 10,
    ) -> float:
        """Find Maximum Fillet Size

        Find the largest fillet radius for the given Shape and edges with a
        recursive binary search.

        Example:

              max_fillet_radius = my_shape.max_fillet(shape_edges)
              max_fillet_radius = my_shape.max_fillet(shape_edges, tolerance=0.5, max_iterations=8)


        Args:
            edge_list (Iterable[Edge]): a sequence of Edge objects, which must belong to this solid
            tolerance (float, optional): maximum error from actual value. Defaults to 0.1.
            max_iterations (int, optional): maximum number of recursive iterations. Defaults to 10.

        Raises:
            RuntimeError: failed to find the max value
            ValueError: the provided Shape is invalid

        Returns:
            float: maximum fillet radius
        """

        def __max_fillet(window_min: float, window_max: float, current_iteration: int):
            window_mid = (window_min + window_max) / 2

            if current_iteration == max_iterations:
                raise RuntimeError(
                    f"Failed to find the max value within {tolerance} in {max_iterations}"
                )

            fillet_builder = BRepFilletAPI_MakeFillet(self.wrapped)

            for native_edge in native_edges:
                fillet_builder.Add(window_mid, native_edge)

            # Do these numbers work? - if not try with the smaller window
            try:
                new_shape = self._make_3d_result(fillet_builder.Shape())
                if not new_shape.is_valid:
                    # raise fillet_exception
                    raise Standard_Failure
            # except fillet_exception:
            except (Standard_Failure, StdFail_NotDone):
                return __max_fillet(window_min, window_mid, current_iteration + 1)

            # These numbers work, are they close enough? - if not try larger window
            if window_mid - window_min <= tolerance:
                return_value = window_mid
            else:
                return_value = __max_fillet(
                    window_mid, window_max, current_iteration + 1
                )
            return return_value

        if not self.is_valid:
            raise ValueError("Invalid Shape")

        native_edges = [e.wrapped for e in edge_list]

        # Unfortunately, MacOS doesn't support the StdFail_NotDone exception so platform
        # specific exceptions are required.
        # if platform.system() == "Darwin":
        #     fillet_exception = Standard_Failure
        # else:
        #     fillet_exception = StdFail_NotDone

        max_radius = __max_fillet(0.0, 2 * self.bounding_box().diagonal, 0)

        return max_radius

    def offset_3d(
        self,
        openings: Iterable[Face] | None,
        thickness: float,
        tolerance: float = 0.0001,
        kind: Kind = Kind.ARC,
    ) -> Solid:
        """Shell

        Make an offset solid of self.

        Args:
            openings (Optional[Iterable[Face]]): faces to be removed,
                which must be part of the solid. Can be an empty list.
            thickness (float): offset amount - positive offset outwards, negative inwards
            tolerance (float, optional): modelling tolerance of the method. Defaults to 0.0001.
            kind (Kind, optional): intersection type. Defaults to Kind.ARC.

        Raises:
            ValueError: Kind.TANGENT not supported

        Returns:
            Solid: A shelled solid.
        """
        openings = list(openings) if openings else []
        if kind == Kind.TANGENT:
            raise ValueError("Kind.TANGENT not supported")

        kind_dict = {
            Kind.ARC: GeomAbs_JoinType.GeomAbs_Arc,
            Kind.INTERSECTION: GeomAbs_JoinType.GeomAbs_Intersection,
            Kind.TANGENT: GeomAbs_JoinType.GeomAbs_Tangent,
        }

        occ_faces_list = TopTools_ListOfShape()
        for face in openings:
            occ_faces_list.Append(face.wrapped)

        offset_builder = BRepOffsetAPI_MakeThickSolid()
        offset_builder.MakeThickSolidByJoin(
            self.wrapped,
            occ_faces_list,
            thickness,
            tolerance,
            Intersection=True,
            RemoveIntEdges=True,
            Join=kind_dict[kind],
        )
        offset_builder.Build()

        try:
            offset_occt_solid = offset_builder.Shape()
        except (StdFail_NotDone, Standard_Failure) as err:
            raise RuntimeError(
                "offset Error, an alternative kind may resolve this error"
            ) from err

        offset_solid = self.__class__.cast(offset_occt_solid)
        assert offset_solid.wrapped is not None

        # The Solid can be inverted, if so reverse
        if offset_solid.volume < 0:
            offset_solid.wrapped.Reverse()

        return offset_solid

    def project_to_viewport(
        self,
        viewport_origin: VectorLike,
        viewport_up: VectorLike = (0, 0, 1),
        look_at: VectorLike | None = None,
        focus: float | None = None,
    ) -> tuple[ShapeList[Edge], ShapeList[Edge]]:
        """project_to_viewport

        Project a shape onto a viewport returning visible and hidden Edges.

        Args:
            viewport_origin (VectorLike): location of viewport
            viewport_up (VectorLike, optional): direction of the viewport y axis.
                Defaults to (0, 0, 1).
            look_at (VectorLike, optional): point to look at.
                Defaults to None (center of shape).
            focus (float, optional): the focal length for perspective projection
                Defaults to None (orthographic projection)

        Returns:
            tuple[ShapeList[Edge],ShapeList[Edge]]: visible & hidden Edges
        """
        return Mixin1D.project_to_viewport(
            self, viewport_origin, viewport_up, look_at, focus
        )


class Solid(Mixin3D[TopoDS_Solid]):
    """A Solid in build123d represents a three-dimensional solid geometry
    in a topological structure. A solid is a closed and bounded volume, enclosing
    a region in 3D space. It comprises faces, edges, and vertices connected in a
    well-defined manner. Solid modeling operations, such as Boolean
    operations (union, intersection, and difference), are often performed on
    Solid objects to create or modify complex geometries."""

    order = 3.0
    # ---- Constructor ----

    def __init__(
        self,
        obj: TopoDS_Solid | Shell | None = None,
        label: str = "",
        color: Color | None = None,
        material: str = "",
        joints: dict[str, Joint] | None = None,
        parent: Compound | None = None,
    ):
        """Build a solid from an OCCT TopoDS_Shape/TopoDS_Solid

        Args:
            obj (TopoDS_Shape | Shell, optional): OCCT Solid or Shell.
            label (str, optional): Defaults to ''.
            color (Color, optional): Defaults to None.
            material (str, optional): tag for external tools. Defaults to ''.
            joints (dict[str, Joint], optional): names joints. Defaults to None.
            parent (Compound, optional): assembly parent. Defaults to None.
        """

        if isinstance(obj, Shell):
            obj = Solid._make_solid(obj)

        super().__init__(
            obj=obj,
            # label="" if label is None else label,
            label=label,
            color=color,
            parent=parent,
        )
        self.material = "" if material is None else material
        self.joints = {} if joints is None else joints

    # ---- Properties ----

    @property
    def volume(self) -> float:
        """volume - the volume of this Solid"""
        # when density == 1, mass == volume
        return Shape.compute_mass(self)

    # ---- Instance Methods ----

    def touch(
        self,
        other: Shape,
        tolerance: float = 1e-6,
        found_solids: ShapeList | None = None,
    ) -> ShapeList[Vertex | Edge | Face]:
        """Find where this Solid's boundary contacts another shape.

        Returns geometry where boundaries contact without interior overlap:
        - Solid + Solid → Face + Edge + Vertex (all boundary contacts)
        - Solid + Face/Shell → Face + Edge + Vertex (boundary contacts)
        - Solid + Edge/Wire → Vertex (edge endpoints on solid boundary)
        - Solid + Vertex → Vertex if on boundary
        - Solid + Compound → distributes over compound elements

        Args:
            other: Shape to check boundary contacts with
            tolerance: tolerance for contact detection
            found_solids: pre-found intersection solids to filter against

        Returns:
            ShapeList of boundary contact geometry (empty if no contact)
        """

        # Helper functions for common geometric checks (for readability)
        # Single shape versions for checking against one shapes
        def vertex_on_edge(v: Vertex, e: Edge) -> bool:
            return v.distance_to(e) <= tolerance

        def vertex_on_face(v: Vertex, f: Face) -> bool:
            return v.distance_to(f) <= tolerance

        def edge_on_face(e: Edge, f: Face) -> bool:
            # Can't use distance_to (e.g. normal vector would match), need Common
            return bool(self._bool_op_list((e,), (f,), BRepAlgoAPI_Common()))

        # Multi shape versions for checking against multiple shapes
        def vertex_on_edges(v: Vertex, edges: Iterable[Edge]) -> bool:
            return any(vertex_on_edge(v, e) for e in edges)

        def vertex_on_faces(v: Vertex, faces: Iterable[Face]) -> bool:
            return any(vertex_on_face(v, f) for f in faces)

        def edge_on_faces(e: Edge, faces: Iterable[Face]) -> bool:
            return any(edge_on_face(e, f) for f in faces)

        def face_point_normal(face: Face, u: float, v: float) -> tuple[Vector, Vector]:
            """Get both position and normal at UV coordinates.
            Args
                u (float): the horizontal coordinate in the parameter space of the Face,
                    between 0.0 and 1.0
                v (float): the vertical coordinate in the parameter space of the Face,
                    between 0.0 and 1.0
            Returns:
                tuple[Vector, Vector]: [point on Face, normal at point]
            """
            u0, u1, v0, v1 = face._uv_bounds()
            u_val = u0 + u * (u1 - u0)
            v_val = v0 + v * (v1 - v0)
            gp_pnt = gp_Pnt()
            gp_norm = gp_Vec()
            BRepGProp_Face(face.wrapped).Normal(u_val, v_val, gp_pnt, gp_norm)
            return Vector(gp_pnt), Vector(gp_norm)

        def faces_equal(f1: Face, f2: Face, grid_size: int = 4) -> bool:
            """Check if two faces are geometrically equal.

            Face == uses topological equality (same OCC object), but we need
            geometric equality. For performance reasons apply a heuristic
            approach: Compare a grid of UV sample points, checking both position and
            normal direction match within tolerance.
            """
            # Early reject: bounding box check
            bb1 = f1.bounding_box(optimal=False)
            bb2 = f2.bounding_box(optimal=False)
            if not bb1.overlaps(bb2, tolerance):
                return False

            # Compare grid_size x grid_size grid of points in UV space
            for i in range(grid_size):
                u = i / (grid_size - 1)
                for j in range(grid_size):
                    v = j / (grid_size - 1)
                    pos1, norm1 = face_point_normal(f1, u, v)
                    pos2, norm2 = face_point_normal(f2, u, v)
                    if (pos1 - pos2).length > tolerance or abs(norm1.dot(norm2)) < 0.99:
                        return False
            return True

        def is_duplicate(shape: Shape, existing: Iterable[Shape]) -> bool:
            if isinstance(shape, Vertex):
                return any(
                    isinstance(v, Vertex) and Vector(shape) == Vector(v)
                    for v in existing
                )
            if isinstance(shape, Edge):
                return any(
                    isinstance(e, Edge) and shape.geom_equal(e, tolerance)
                    for e in existing
                )
            if isinstance(shape, Face):
                # Heuristic approach
                return any(
                    isinstance(f, Face) and faces_equal(shape, f) for f in existing
                )
            return False

        results: ShapeList = ShapeList()

        if isinstance(other, (Solid, Face, Shell)):
            # Unified handling: iterate over face pairs
            # For Solid+Solid: get intersection solids to filter results that bound them
            intersect_faces = []
            if isinstance(other, Solid):
                if found_solids is None:
                    found_solids = ShapeList(
                        self._intersect(other, tolerance, include_touched=False) or []
                    )
                intersect_faces = [f for s in found_solids for f in s.faces()]

            # Pre-calculate bounding boxes for early rejection
            self_faces = [(f, f.bounding_box(optimal=False)) for f in self.faces()]
            other_faces = [(f, f.bounding_box(optimal=False)) for f in other.faces()]

            # First pass: collect touch/intersect results from face pairs,
            # filtering against intersection solid faces
            raw_results: ShapeList = ShapeList()
            for sf, sf_bb in self_faces:
                for of, of_bb in other_faces:
                    if not sf_bb.overlaps(of_bb, tolerance):
                        continue

                    # Process touch first (cheap), then intersect (expensive)
                    # Face touch gives tangent vertices
                    for r in sf.touch(of, tolerance=tolerance):
                        if not is_duplicate(r, raw_results) and not vertex_on_faces(
                            r, intersect_faces
                        ):
                            raw_results.append(r)

                    # Face intersect gives shared faces/edges (touch handled above)
                    for r in sf.intersect(of, tolerance=tolerance) or []:
                        if not is_duplicate(r, raw_results) and not edge_on_faces(
                            r, intersect_faces
                        ):
                            raw_results.append(r)

            # Second pass: filter lower-dimensional results against higher-dimensional
            all_faces = [f for f in raw_results if isinstance(f, Face)]
            all_edges = [e for e in raw_results if isinstance(e, Edge)]
            for r in raw_results:
                if (
                    isinstance(r, Face)
                    or (isinstance(r, Edge) and not edge_on_faces(r, all_faces))
                    or (
                        isinstance(r, Vertex)
                        and not vertex_on_faces(r, all_faces)
                        and not vertex_on_edges(r, all_edges)
                    )
                ):
                    results.append(r)

        elif isinstance(other, (Edge, Wire)):
            # Solid + Edge: find where edge endpoints touch solid boundary
            # Pre-calculate bounding boxes (optimal=False for speed, used for filtering)
            self_faces = [(f, f.bounding_box(optimal=False)) for f in self.faces()]
            other_bb = other.bounding_box(optimal=False)

            for ov in other.vertices():
                for sf, _ in self_faces:
                    if vertex_on_face(ov, sf):
                        results.append(ov)
                        break

            # Use BRepExtrema to find all tangent contacts (edge tangent to surface)
            for sf, sf_bb in self_faces:
                if not sf_bb.overlaps(other_bb, tolerance):
                    continue
                extrema = BRepExtrema_DistShapeShape(sf.wrapped, other.wrapped)
                if extrema.IsDone() and extrema.Value() <= tolerance:
                    for i in range(1, extrema.NbSolution() + 1):
                        pnt1 = extrema.PointOnShape1(i)
                        pnt2 = extrema.PointOnShape2(i)
                        if pnt1.Distance(pnt2) <= tolerance:
                            new_vertex = Vertex(pnt1.X(), pnt1.Y(), pnt1.Z())
                            if not is_duplicate(new_vertex, results):
                                results.append(new_vertex)

        elif isinstance(other, Vertex):
            # Solid + Vertex: check if vertex is on boundary
            for sf in self.faces():
                if vertex_on_face(other, sf):
                    results.append(other)
                    break

        # Delegate to other shapes (Compound iterates, others return empty)
        else:
            results.extend(other.touch(self, tolerance))

        # Remove duplicates using Shape's __hash__ and __eq__
        return ShapeList(set(results))

    # ---- Class Methods ----

    @classmethod
    def _make_solid(cls, shell: Shell) -> TopoDS_Solid:
        """Create a Solid object from the surface shell"""
        return ShapeFix_Solid().SolidFromShell(shell.wrapped)

    @classmethod
    def _set_sweep_mode(
        cls,
        builder: BRepOffsetAPI_MakePipeShell,
        path: Wire | Edge,
        binormal: Vector | Wire | Edge,
    ) -> bool:
        rotate = False

        if isinstance(binormal, Vector):
            coordinate_system = gp_Ax2()
            coordinate_system.SetLocation(path.start_point().to_pnt())
            coordinate_system.SetDirection(binormal.to_dir())
            builder.SetMode(coordinate_system)
            rotate = True
        elif isinstance(binormal, (Wire, Edge)):
            builder.SetMode(Wire(binormal).wrapped, True)

        return rotate

    @classmethod
    def extrude(cls, obj: Face, direction: VectorLike) -> Solid:
        """extrude

        Extrude a Face into a Solid.

        Args:
            direction (VectorLike): direction and magnitude of extrusion

        Raises:
            ValueError: Unsupported class
            RuntimeError: Generated invalid result

        Returns:
            Edge: extruded shape
        """
        return Solid(TopoDS.Solid(_extrude_topods_shape(obj.wrapped, direction)))

    @classmethod
    def extrude_linear_with_rotation(
        cls,
        section: Face | Wire,
        center: VectorLike,
        normal: VectorLike,
        angle: float,
        inner_wires: list[Wire] | None = None,
    ) -> Solid:
        """Extrude with Rotation

        Creates a 'twisted prism' by extruding, while simultaneously rotating around the
        extrusion vector.

        Args:
            section (Union[Face,Wire]): cross section
            vec_center (VectorLike): the center point about which to rotate
            vec_normal (VectorLike): a vector along which to extrude the wires
            angle (float): the angle to rotate through while extruding
            inner_wires (list[Wire], optional): holes - only used if section is of type Wire.
                Defaults to None.

        Returns:
            Solid: extruded object
        """
        # Though the signature may appear to be similar enough to extrude to merit
        # combining them, the construction methods used here are different enough that they
        # should be separate.

        # At a high level, the steps followed are:
        # (1) accept a set of wires
        # (2) create another set of wires like this one, but which are transformed and rotated
        # (3) create a ruledSurface between the sets of wires
        # (4) create a shell and compute the resulting object

        inner_wires = inner_wires if inner_wires else []
        center = Vector(center)
        normal = Vector(normal)

        def extrude_aux_spine(
            wire: TopoDS_Wire, spine: TopoDS_Wire, aux_spine: TopoDS_Wire
        ) -> TopoDS_Shape:
            """Helper function"""
            extrude_builder = BRepOffsetAPI_MakePipeShell(spine)
            extrude_builder.SetMode(aux_spine, False)  # auxiliary spine
            extrude_builder.Add(wire)
            extrude_builder.Build()
            extrude_builder.MakeSolid()
            return extrude_builder.Shape()

        if isinstance(section, Face):
            outer_wire = section.outer_wire()
            inner_wires = section.inner_wires()
        else:
            outer_wire = section

        # make straight spine
        straight_spine_e = Edge.make_line(center, center.add(normal))
        straight_spine_wires = Wire.combine([straight_spine_e])
        straight_spine_w = straight_spine_wires[0].wrapped  # pylint: disable=no-member

        # make an auxiliary spine
        pitch = 360.0 / angle * normal.length
        aux_spine_w = Wire(
            [Edge.make_helix(pitch, normal.length, 1, center=center, normal=normal)]
        ).wrapped

        # extrude the outer wire
        outer_solid = extrude_aux_spine(
            outer_wire.wrapped, straight_spine_w, aux_spine_w
        )

        # extrude inner wires
        inner_solids = [
            extrude_aux_spine(w.wrapped, straight_spine_w, aux_spine_w)
            for w in inner_wires
        ]

        # combine the inner solids into compound
        inner_comp = _make_topods_compound_from_shapes(inner_solids)

        # subtract from the outer solid
        difference = BRepAlgoAPI_Cut(outer_solid, inner_comp).Shape()

        # convert to a TopoDS_Solid - might be wrapped in a TopoDS_Compound
        try:
            result = TopoDS.Solid(difference)
        except Standard_TypeMismatch:
            result = TopoDS.Solid(
                unwrap_topods_compound(TopoDS.Compound(difference), True)
            )

        return Solid(result)

    @classmethod
    def extrude_taper(
        cls, profile: Face, direction: VectorLike, taper: float, flip_inner: bool = True
    ) -> Solid:
        """Extrude a cross section with a taper

        Extrude a cross section into a prismatic solid in the provided direction.

        Note that two difference algorithms are used. If direction aligns with
        the profile normal (which must be positive), the taper is positive and the profile
        contains no holes the OCP LocOpe_DPrism algorithm is used as it generates the most
        accurate results. Otherwise, a loft is created between the profile and the profile
        with a 2D offset set at the appropriate direction.

        Args:
            section (Face]): cross section
            normal (VectorLike): a vector along which to extrude the wires. The length
                of the vector controls the length of the extrusion.
            taper (float): taper angle in degrees.
            flip_inner (bool, optional): outer and inner geometry have opposite tapers to
                allow for part extraction when injection molding.

        Returns:
            Solid: extruded cross section
        """
        # pylint: disable=too-many-locals
        direction = Vector(direction)

        if (
            direction.normalized() == profile.normal_at()
            and Plane(profile).z_dir.Z > 0
            and taper > 0
            and not profile.inner_wires()
        ):
            prism_builder = LocOpe_DPrism(
                profile.wrapped,
                direction.length / cos(radians(taper)),
                radians(taper),
            )
            new_solid = Solid(TopoDS.Solid(prism_builder.Shape()))
        else:
            # Determine the offset to get the taper
            offset_amt = -direction.length * tan(radians(taper))

            outer = profile.outer_wire()
            local_outer: Wire = Plane(profile).to_local_coords(outer)
            local_taper_outer = local_outer.offset_2d(
                offset_amt, kind=Kind.INTERSECTION
            )
            taper_outer = Plane(profile).from_local_coords(local_taper_outer)
            taper_outer.move(Location(direction))

            profile_wires = [profile.outer_wire()] + profile.inner_wires()

            taper_wires = []
            for i, wire in enumerate(profile_wires):
                flip = -1 if i > 0 and flip_inner else 1
                local: Wire = Plane(profile).to_local_coords(wire)
                local_taper = local.offset_2d(flip * offset_amt, kind=Kind.INTERSECTION)
                taper_wire: Wire = Plane(profile).from_local_coords(local_taper)
                taper_wire.move(Location(direction))
                taper_wires.append(taper_wire)

            solids = [
                Solid.make_loft([p, t]) for p, t in zip(profile_wires, taper_wires)
            ]
            if len(solids) > 1:
                complex_solid = solids[0].cut(*solids[1:])
                assert isinstance(complex_solid, Solid)  # Can't be a list
                new_solid = complex_solid
            else:
                new_solid = solids[0]

        return new_solid

    @classmethod
    def extrude_until(
        cls,
        profile: Face,
        target: Compound | Solid,
        direction: VectorLike,
        until: Until = Until.NEXT,
    ) -> Solid:
        """extrude_until

        Extrude `profile` in the provided `direction` until it encounters a
        bounding surface on the `target`. The termination surface is chosen
        according to the `until` option:

            * ``Until.NEXT`` — Extrude forward until the first intersecting surface.
            * ``Until.LAST`` — Extrude forward through all intersections, stopping at
            the farthest surface.
            * ``Until.PREVIOUS`` — Reverse the extrusion direction and stop at the
            first intersecting surface behind the profile.
            * ``Until.FIRST`` — Reverse the direction and stop at the farthest
            surface behind the profile.

        When ``Until.PREVIOUS`` or ``Until.FIRST`` are used, the extrusion
        direction is automatically inverted before execution.

        Note:
            The bounding surface on the target must be large enough to
            completely cover the extruded profile at the contact region.
            Partial overlaps may yield open or invalid solids.

        Args:
            profile (Face): The face to extrude.
            target (Union[Compound, Solid]): The object that limits the extrusion.
            direction (VectorLike): Extrusion direction.
            until (Until, optional): Surface selection mode controlling which
                intersection to stop at. Defaults to ``Until.NEXT``.

        Raises:
            ValueError: If the provided profile does not intersect the target.

        Returns:
            Solid: The extruded and limited solid.
        """
        direction = Vector(direction)
        if until in [Until.PREVIOUS, Until.FIRST]:
            direction *= -1
            until = Until.NEXT if until == Until.PREVIOUS else Until.LAST

        # 1: Create extrusion of length the maximum distance between profile and target
        max_dimension = find_max_dimension([profile, target])
        extrusion = Solid.extrude(profile, direction * max_dimension)

        # 2: Intersect the extrusion with the target to find the target's modified faces
        intersect_op = BRepAlgoAPI_Common(target.wrapped, extrusion.wrapped)
        intersect_op.Build()
        intersection = intersect_op.Shape()
        face_exp = TopExp_Explorer(intersection, ta.TopAbs_FACE)
        if not face_exp.More():
            raise ValueError("No intersection: extrusion does not contact target")

        # Find the faces from the intersection that originated on the target
        history = intersect_op.History()
        modified_target_faces = []
        face_explorer = TopExp_Explorer(target.wrapped, ta.TopAbs_FACE)
        while face_explorer.More():
            target_face = TopoDS.Face(face_explorer.Current())
            modified_los: TopTools_ListOfShape = history.Modified(target_face)
            while not modified_los.IsEmpty():
                modified_face = TopoDS.Face(modified_los.First())
                modified_los.RemoveFirst()
                modified_target_faces.append(modified_face)
            face_explorer.Next()

        # 3: Sew the resulting faces into shells - one for each surface the extrusion
        #    passes through and sort by distance from the profile
        sewed_shape = _sew_topods_faces(modified_target_faces)

        # From the sewed shape extract the shells and single faces
        top_level_shapes = get_top_level_topods_shapes(sewed_shape)
        modified_target_surfaces: ShapeList[Face | Shell] = ShapeList()

        # For each of the top level Shells and Faces
        for top_level_shape in top_level_shapes:
            if isinstance(top_level_shape, TopoDS_Face):
                modified_target_surfaces.append(Face(top_level_shape))
            elif isinstance(top_level_shape, TopoDS_Shell):
                modified_target_surfaces.append(Shell(top_level_shape))
            else:
                raise RuntimeError(f"Invalid sewn shape {type(top_level_shape)}")

        modified_target_surfaces = modified_target_surfaces.sort_by(
            lambda s: s.distance_to(profile)
        )
        limit = modified_target_surfaces[
            0 if until in [Until.NEXT, Until.PREVIOUS] else -1
        ]
        keep: Literal[Keep.TOP, Keep.BOTTOM] = (
            Keep.TOP if until in [Until.NEXT, Until.PREVIOUS] else Keep.BOTTOM
        )

        # 4: Split the extrusion by the appropriate shell
        clipped_extrusion = extrusion.split(limit, keep=keep)

        # 5: Return the appropriate type
        if clipped_extrusion is None:
            raise RuntimeError("Extrusion is None")  # None isn't an option here
        if isinstance(clipped_extrusion, Solid):
            return clipped_extrusion
        #  isinstance(clipped_extrusion, list):
        return ShapeList(clipped_extrusion).sort_by(Axis(profile.center(), direction))[
            0
        ]

    @classmethod
    def from_bounding_box(cls, bbox: BoundBox | OrientedBoundBox) -> Solid:
        """A box of the same dimensions and location"""
        if isinstance(bbox, BoundBox):
            return Solid.make_box(*bbox.size).locate(Location(bbox.min))
        moved_plane: Plane = Plane(Location(-bbox.size / 2)).moved(bbox.location)
        return Solid.make_box(bbox.size.X, bbox.size.Y, bbox.size.Z, plane=moved_plane)

    @classmethod
    def make_box(
        cls, length: float, width: float, height: float, plane: Plane = Plane.XY
    ) -> Solid:
        """make box

        Make a box at the origin of plane extending in positive direction of each axis.

        Args:
            length (float):
            width (float):
            height (float):
            plane (Plane, optional): base plane. Defaults to Plane.XY.

        Returns:
            Solid: Box
        """
        return cls(
            TopoDS.Solid(
                BRepPrimAPI_MakeBox(
                    plane.to_gp_ax2(),
                    length,
                    width,
                    height,
                ).Shape()
            )
        )

    @classmethod
    def make_cone(
        cls,
        base_radius: float,
        top_radius: float,
        height: float,
        plane: Plane = Plane.XY,
        angle: float = 360,
    ) -> Solid:
        """make cone

        Make a cone with given radii and height

        Args:
            base_radius (float):
            top_radius (float):
            height (float):
            plane (Plane): base plane. Defaults to Plane.XY.
            angle (float, optional): arc size. Defaults to 360.

        Returns:
            Solid: Full or partial cone
        """
        return cls(
            TopoDS.Solid(
                BRepPrimAPI_MakeCone(
                    plane.to_gp_ax2(),
                    base_radius,
                    top_radius,
                    height,
                    angle * DEG2RAD,
                ).Shape()
            )
        )

    @classmethod
    def make_cylinder(
        cls,
        radius: float,
        height: float,
        plane: Plane = Plane.XY,
        angle: float = 360,
    ) -> Solid:
        """make cylinder

        Make a cylinder with a given radius and height with the base center on plane origin.

        Args:
            radius (float):
            height (float):
            plane (Plane): base plane. Defaults to Plane.XY.
            angle (float, optional): arc size. Defaults to 360.

        Returns:
            Solid: Full or partial cylinder
        """
        return cls(
            TopoDS.Solid(
                BRepPrimAPI_MakeCylinder(
                    plane.to_gp_ax2(),
                    radius,
                    height,
                    angle * DEG2RAD,
                ).Shape()
            )
        )

    @classmethod
    def make_loft(cls, objs: Iterable[Vertex | Wire], ruled: bool = False) -> Solid:
        """make loft

        Makes a loft from a list of wires and vertices. Vertices can appear only at the
        beginning or end of the list, but cannot appear consecutively within the list
        nor between wires.

        Args:
            objs (list[Vertex, Wire]): wire perimeters or vertices
            ruled (bool, optional): stepped or smooth. Defaults to False (smooth).

        Raises:
            ValueError: Too few wires

        Returns:
            Solid: Lofted object
        """
        return cls(TopoDS.Solid(_make_loft(objs, True, ruled)))

    @classmethod
    def make_sphere(
        cls,
        radius: float,
        plane: Plane = Plane.XY,
        angle1: float = -90,
        angle2: float = 90,
        angle3: float = 360,
    ) -> Solid:
        """Sphere

        Make a full or partial sphere - with a given radius center on the origin or plane.

        Args:
            radius (float):
            plane (Plane): base plane. Defaults to Plane.XY.
            angle1 (float, optional): Defaults to -90.
            angle2 (float, optional): Defaults to 90.
            angle3 (float, optional): Defaults to 360.

        Returns:
            Solid: sphere
        """
        return cls(
            TopoDS.Solid(
                BRepPrimAPI_MakeSphere(
                    plane.to_gp_ax2(),
                    radius,
                    angle1 * DEG2RAD,
                    angle2 * DEG2RAD,
                    angle3 * DEG2RAD,
                ).Shape()
            )
        )

    @classmethod
    def make_torus(
        cls,
        major_radius: float,
        minor_radius: float,
        plane: Plane = Plane.XY,
        start_angle: float = 0,
        end_angle: float = 360,
        major_angle: float = 360,
    ) -> Solid:
        """make torus

        Make a torus with a given radii and angles

        Args:
            major_radius (float):
            minor_radius (float):
            plane (Plane): base plane. Defaults to Plane.XY.
            start_angle (float, optional): start major arc. Defaults to 0.
            end_angle (float, optional): end major arc. Defaults to 360.

        Returns:
            Solid: Full or partial torus
        """
        return cls(
            TopoDS.Solid(
                BRepPrimAPI_MakeTorus(
                    plane.to_gp_ax2(),
                    major_radius,
                    minor_radius,
                    start_angle * DEG2RAD,
                    end_angle * DEG2RAD,
                    major_angle * DEG2RAD,
                ).Shape()
            )
        )

    @classmethod
    def make_wedge(
        cls,
        delta_x: float,
        delta_y: float,
        delta_z: float,
        min_x: float,
        min_z: float,
        max_x: float,
        max_z: float,
        plane: Plane = Plane.XY,
    ) -> Solid:
        """Make a wedge

        Args:
            delta_x (float):
            delta_y (float):
            delta_z (float):
            min_x (float):
            min_z (float):
            max_x (float):
            max_z (float):
            plane (Plane): base plane. Defaults to Plane.XY.

        Returns:
            Solid: wedge
        """
        return cls(
            TopoDS.Solid(
                BRepPrimAPI_MakeWedge(
                    plane.to_gp_ax2(),
                    delta_x,
                    delta_y,
                    delta_z,
                    min_x,
                    min_z,
                    max_x,
                    max_z,
                ).Solid()
            )
        )

    @classmethod
    def revolve(
        cls,
        section: Face | Wire,
        angle: float,
        axis: Axis,
        inner_wires: list[Wire] | None = None,
    ) -> Solid:
        """Revolve

        Revolve a cross section about the given Axis by the given angle.

        Args:
            section (Union[Face,Wire]): cross section
            angle (float): the angle to revolve through
            axis (Axis): rotation Axis
            inner_wires (list[Wire], optional): holes - only used if section is of type Wire.
                Defaults to [].

        Returns:
            Solid: the revolved cross section
        """
        inner_wires = inner_wires if inner_wires else []
        if isinstance(section, Wire):
            section_face = Face(section, inner_wires)
        else:
            section_face = section

        revol_builder = BRepPrimAPI_MakeRevol(
            section_face.wrapped,
            axis.wrapped,
            angle * DEG2RAD,
            True,
        )

        return cls(TopoDS.Solid(revol_builder.Shape()))

    @classmethod
    def sweep(
        cls,
        section: Face | Wire,
        path: Wire | Edge,
        inner_wires: list[Wire] | None = None,
        make_solid: bool = True,
        is_frenet: bool = False,
        mode: Vector | Wire | Edge | None = None,
        transition: Transition = Transition.TRANSFORMED,
    ) -> Solid:
        """Sweep

        Sweep the given cross section into a prismatic solid along the provided path

        The is_frenet parameter controls how the profile orientation changes as it
        follows along the sweep path. If is_frenet is False, the orientation of the
        profile is kept consistent from point to point. The resulting shape has the
        minimum possible twisting. Unintuitively, when a profile is swept along a
        helix, this results in the orientation of the profile slowly creeping
        (rotating) as it follows the helix. Setting is_frenet to True prevents this.

        If is_frenet is True the orientation of the profile is based on the local
        curvature and tangency vectors of the path. This keeps the orientation of the
        profile consistent when sweeping along a helix (because the curvature vector of
        a straight helix always points to its axis). However, when path is not a helix,
        the resulting shape can have strange looking twists sometimes. For more
        information, see Frenet Serret formulas
        http://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas.

        Args:
            section (Union[Face, Wire]): cross section to sweep
            path (Union[Wire, Edge]): sweep path
            inner_wires (list[Wire]): holes - only used if section is a wire
            make_solid (bool, optional): return Solid or Shell. Defaults to True.
            is_frenet (bool, optional): Frenet mode. Defaults to False.
            mode (Union[Vector, Wire, Edge, None], optional): additional sweep
                mode parameters. Defaults to None.
            transition (Transition, optional): handling of profile orientation at C1 path
                discontinuities. Defaults to Transition.TRANSFORMED.

        Returns:
            Solid: the swept cross section
        """
        if isinstance(section, Face):
            outer_wire = section.outer_wire()
            inner_wires = section.inner_wires()
        else:
            outer_wire = section
            inner_wires = inner_wires if inner_wires else []

        shapes: list[Mixin3D[TopoDS_Shape]] = []
        for wire in [outer_wire] + inner_wires:
            builder = BRepOffsetAPI_MakePipeShell(Wire(path).wrapped)

            rotate = False

            # handle sweep mode
            if mode:
                rotate = Solid._set_sweep_mode(builder, path, mode)
            else:
                builder.SetMode(is_frenet)

            builder.SetTransitionMode(Shape._transModeDict[transition])

            builder.Add(wire.wrapped, False, rotate)

            builder.Build()
            if make_solid:
                builder.MakeSolid()

            shapes.append(Mixin3D.cast(builder.Shape()))

        outer_shape, inner_shapes = shapes[0], shapes[1:]

        if inner_shapes:
            hollow_outer_shape = outer_shape.cut(*inner_shapes)
            assert isinstance(hollow_outer_shape, Solid)
            return hollow_outer_shape

        return outer_shape

    @classmethod
    def sweep_multi(
        cls,
        profiles: Iterable[Wire | Face],
        path: Wire | Edge,
        make_solid: bool = True,
        is_frenet: bool = False,
        binormal: Vector | Wire | Edge | None = None,
    ) -> Solid:
        """Multi section sweep

        Sweep through a sequence of profiles following a path.

        The is_frenet parameter controls how the profile orientation changes as it
        follows along the sweep path. If is_frenet is False, the orientation of the
        profile is kept consistent from point to point. The resulting shape has the
        minimum possible twisting. Unintuitively, when a profile is swept along a
        helix, this results in the orientation of the profile slowly creeping
        (rotating) as it follows the helix. Setting is_frenet to True prevents this.

        If is_frenet is True the orientation of the profile is based on the local
        curvature and tangency vectors of the path. This keeps the orientation of the
        profile consistent when sweeping along a helix (because the curvature vector of
        a straight helix always points to its axis). However, when path is not a helix,
        the resulting shape can have strange looking twists sometimes. For more
        information, see Frenet Serret formulas
        http://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas.

        Args:
            profiles (Iterable[Union[Wire, Face]]): list of profiles
            path (Union[Wire, Edge]): The wire to sweep the face resulting from the wires over
            make_solid (bool, optional): Solid or Shell. Defaults to True.
            is_frenet (bool, optional): Select frenet mode. Defaults to False.
            binormal (Union[Vector, Wire, Edge, None], optional): additional sweep mode parameters.
                Defaults to None.

        Returns:
            Solid: swept object
        """
        path_as_wire = Wire(path).wrapped

        builder = BRepOffsetAPI_MakePipeShell(path_as_wire)

        translate = False
        rotate = False

        if binormal:
            rotate = cls._set_sweep_mode(builder, path, binormal)
        else:
            builder.SetMode(is_frenet)

        for profile in profiles:
            path_as_wire = (
                profile.wrapped
                if isinstance(profile, Wire)
                else profile.outer_wire().wrapped
            )
            builder.Add(path_as_wire, translate, rotate)

        builder.Build()

        if make_solid:
            builder.MakeSolid()

        return cls(TopoDS.Solid(builder.Shape()))

    @classmethod
    def thicken(
        cls,
        surface: Face | Shell,
        depth: float,
        normal_override: VectorLike | None = None,
    ) -> Solid:
        """Thicken Face or Shell

        Create a solid from a potentially non planar face or shell by thickening along
        the normals.

        .. image:: thickenFace.png

        Non-planar faces are thickened both towards and away from the center of the sphere.

        Args:
            depth (float): Amount to thicken face(s), can be positive or negative.
            normal_override (Vector, optional): Face only. The normal_override vector can be
                used to indicate which way is 'up', potentially flipping the face normal
                direction such that many faces with different normals all go in the same
                direction (direction need only be +/- 90 degrees from the face normal).
                Defaults to None.

        Raises:
            RuntimeError: Opencascade internal failures

        Returns:
            Solid: The resulting Solid object
        """
        # Check to see if the normal needs to be flipped
        adjusted_depth = depth
        if isinstance(surface, Face) and normal_override is not None:
            surface_center = surface.center()
            surface_normal = surface.normal_at(surface_center).normalized()
            if surface_normal.dot(Vector(normal_override).normalized()) < 0:
                adjusted_depth = -depth

        offset_builder = BRepOffset_MakeOffset()
        offset_builder.Initialize(
            surface.wrapped,
            Offset=adjusted_depth,
            Tol=1.0e-5,
            Mode=BRepOffset_Skin,
            # BRepOffset_RectoVerso - which describes the offset of a given surface shell along both
            # sides of the surface but doesn't seem to work
            Intersection=True,
            SelfInter=False,
            Join=GeomAbs_Intersection,  # Could be GeomAbs_Arc,GeomAbs_Tangent,GeomAbs_Intersection
            Thickening=True,
            RemoveIntEdges=True,
        )
        offset_builder.MakeOffsetShape()
        try:
            result = Solid(TopoDS.Solid(offset_builder.Shape()))
        except StdFail_NotDone as err:
            raise RuntimeError("Error applying thicken to given surface") from err

        return result

    def draft(self, faces: Iterable[Face], neutral_plane: Plane, angle: float) -> Solid:
        """Apply a draft angle to the given faces of the solid.

        Args:
            faces: Faces to which the draft should be applied.
            neutral_plane: Plane defining the neutral direction and position.
            angle: Draft angle in degrees.

        Returns:
            Solid with the specified draft angles applied.

        Raises:
            RuntimeError: If draft application fails on any face or during build.
        """
        valid_geom_types = {GeomType.PLANE, GeomType.CYLINDER, GeomType.CONE}
        for face in faces:
            if face.geom_type not in valid_geom_types:
                raise ValueError(
                    f"Face {face} has unsupported geometry type {face.geom_type.name}. "
                    "Only PLANAR, CYLINDRICAL, and CONICAL faces are supported."
                )

        draft_angle_builder = BRepOffsetAPI_DraftAngle(self.wrapped)

        for face in faces:
            draft_angle_builder.Add(
                face.wrapped,
                neutral_plane.z_dir.to_dir(),
                radians(angle),
                neutral_plane.wrapped,
                Flag=True,
            )
            if not draft_angle_builder.AddDone():
                raise DraftAngleError(
                    "Draft could not be added to a face.",
                    face=face,
                    problematic_shape=draft_angle_builder.ProblematicShape(),
                )

        try:
            draft_angle_builder.Build()
            result = Solid(TopoDS.Solid(draft_angle_builder.Shape()))
        except StdFail_NotDone as err:
            raise DraftAngleError(
                "Draft build failed on the given solid.",
                face=None,
                problematic_shape=draft_angle_builder.ProblematicShape(),
            ) from err
        return result


class DraftAngleError(RuntimeError):
    """Solid.draft custom exception"""

    def __init__(self, message, face=None, problematic_shape=None):
        super().__init__(message)
        self.face = face
        self.problematic_shape = problematic_shape

def compute_unbend_transforms(bend_sequences: List[List[Face]], estimated_thickness: float, adj_graph: nx.Graph, seam_edges) -> None:
    # todo, 
    # return (t_r, r, t_c) 
    # t_r - translation to match the seam edges of the flanges,
    # r - rotate flange by bend angle
    # t_c - bend compensation using k-factor (material dependent)
    
    # each of the 

    def flange_face_rotation(lcs: Plane, child_rot_seams: ShapeList[Edge], parent_face_bend_seams: ShapeList[Edge], bend_angle: float) -> Tuple[Location, float]:
        to_local = Location(lcs).inverse()
        to_world = Location(lcs)

        closest_child_rot_seam = child_rot_seams.sort_by_distance(lcs.origin)[0]
        ref_face_bend_seam = parent_face_bend_seams.sort_by_distance(lcs.origin)[0]

        child_seam_translation: Vector = (to_local * closest_child_rot_seam).center() - (to_local * ref_face_bend_seam).center()
        transformation = Rot(bend_angle, 0, 0) * Pos(0, -child_seam_translation.Y, -child_seam_translation.Z)

        return to_world * transformation * to_local
    
    def bend_allowance_translation(lcs: Plane, bend_radius: float, bend_angle: float, thickness: float, non_seam_cyl_edges: ShapeList[Edge] | None = None, k_factor: float | None = None) -> Location:
        to_local = Location(lcs).inverse()
        to_world = Location(lcs)

        bac = BendAllowanceCalculator.read_file()
        if k_factor is None:
            k_factor = bac.get_k_factor(bend_radius, thickness)
        bend_allowance = (bend_radius + k_factor * thickness) * bend_angle

        bend_direction = 1 if bend_angle > 0 else -1

        return to_world * Pos(0, bend_direction * bend_allowance, 0) * to_local 

    done = []
    child: Face 
    bend: Face 
    parent: Face
    for unbend_path in bend_sequences:
        # edges_to_transform = []
        transformations: List[Tuple[Face, Location]] = []
        i = 0
        for parent, bend, child in zip(
            unbend_path[0::2],    # Elements at indices 0, 2, 4, ...
            unbend_path[1::2],    # Elements at indices 1, 3, 5, ...
            unbend_path[2::2]     # Elements at indices 2, 4, 6, ...
        ):
            # Your loop logic here
            # edges_to_transform.extend(filter(lambda e: e not in seam_edges, child.edges()))
            # edges_to_transform.extend(filter(lambda e: e not in seam_edges, bend.edges()))
            # not_seam.extend(filter(lambda e: e not in seam_edges, parent.edges()))

            child_edges_to_transform = list(filter(lambda e: e not in seam_edges, child.edges()))
            cyl_non_seam_edges = list(filter(lambda e: e not in seam_edges, bend.edges()))

            parent_bend_seam: Edge = adj_graph[parent][bend]['label']
            child_bend_seam:Edge = adj_graph[bend][child]['label']

            # Construct local coordinate system (border trihedron)
            local_origin = parent_bend_seam.start_point()  
            local_x = parent_bend_seam.tangent_at(0.5) 
            local_z = parent.normal_at(local_origin)
            
            lcs = Plane(
                    origin=local_origin,
                    x_dir=local_x,
                    z_dir=local_z
            )

            unsigned_bend_angle: float = parent.normal_at().get_angle(
                child.normal_at()
            )
            to_local = Location(lcs).inverse()
            to_world = Location(lcs)

            rotation_direction = (to_local * child).normal_at().cross((to_local * parent).normal_at())
            bend_angle: float = (unsigned_bend_angle * rotation_direction).X

            rotation: Location = flange_face_rotation(lcs, ShapeList([child_bend_seam]), ShapeList([parent_bend_seam]), bend_angle)

            bend_allowance: Location = bend_allowance_translation(lcs, bend.radius, bend_angle * math.pi / 180, estimated_thickness)

            test = rotation * child
            test = bend_allowance * test
            if i == 0:
                transformations.append((child, bend_allowance * rotation))
            else:
                previous = transformations[i-1][1]
                transformations.append((child,  previous *  bend_allowance * rotation))
            i+=1
        transformed: List[Face] = []
        for t in transformations:
            a = t[1] * t[0]
            transformed.append(a)
        done.append(transformed)
    x = 0
    y = 0
def _unfold(solid_to_unfold: Solid, reference_face: Face, material: float) -> Solid:
    """Unfolds a solid given a reference face, on which plane we unfold the other faces 

    Args:
        solid_to_unfold: Solid to unfold,
        reference_face: Face to which the unfold reference plane is constructed

    Returns:
        The unfolded part as a new Solid
    """
    
    tangent_faces_adjacacency_graph = build_graph(solid_to_unfold, reference_face)

    # depth first search tree with reference face as root
    dfs_tree = nx.dfs_tree(tangent_faces_adjacacency_graph, reference_face)
    bend_sequences: List[List[Face]] = []

    first_flange: Face
    bend: Face
    second_flange: Face

    seam_edges = set()
    for u, v in dfs_tree.edges():
        seam_edges.add(tangent_faces_adjacacency_graph[u][v]['label'])
        if dfs_tree.out_degree(v) == 0:
            # it's a leaf
            unfold_path = nx.shortest_path(dfs_tree, reference_face, v)
            is_valid_path = True
            for i in range(0, len(unfold_path) - 2, 2):
                first_flange, bend, second_flange = unfold_path[i], unfold_path[i+1], unfold_path[i+2]                
                if not (isinstance(first_flange.is_planar, Plane) and (bend.is_circular_convex or bend.is_circular_concave) and isinstance(second_flange.is_planar, Plane)):
                    is_valid_path = False

            if is_valid_path:
                bend_sequences.append(unfold_path)
            else:
               raise RuntimeError(f"Invalid pattern at indices {i}-{i+2}: expected flange -> bend -> flange")
    estimated_thickness = estimate_thickness(solid_to_unfold, reference_face)
    compute_unbend_transforms(bend_sequences, estimated_thickness, tangent_faces_adjacacency_graph, seam_edges)

    
    """ unfolded_flanges, bocklines = unbend_transforms(bend_sequences, thickness)
    start_flans = bend_sequences[0].parent_flange

    unfolded_product = start_flans
    for flange in unfolded_flanges:
        unfolded_product += flange

    bockar = []
    for l in bocklines: 
        bock = Edge.make_line(l[0], l[1])
        bockar.append(bock)
    """
    unfolded_product = None
    bockar = None
    final_component = Compound([unfolded_product] + bockar)

    if final_component:
        show(final_component)

    return final_component 

def build_graph(solid: Solid, root_face: Face) -> nx.Graph:
    adjacent_faces_graph = nx.Graph()
    for i, face in enumerate(solid.faces()):
        # if face.is_circular_concave or face.is_circular_convex:
        if face.geom_type is GeomType.CYLINDER:
            face_edges = face.edges()
            for f_edge in face_edges:
                connected_faces: ShapeList[Face] = ShapeList(
                    map(lambda f: Face(f), topo_explore_connected_faces(f_edge))
                    )
                if len(connected_faces) == 2 and faces_are_tangent(first=connected_faces[0], second=connected_faces[1], common_edge=f_edge):
                    face_1: Face = connected_faces[0]
                    face_2: Face = connected_faces[1]
                    if (face_1.geom_type is GeomType.CYLINDER and isinstance(face_2.is_planar, Plane)) or \
                        (face_2.geom_type is GeomType.CYLINDER and isinstance(face_1.is_planar, Plane)):
                        
                        adjacent_faces_graph.add_node(face_1, type=face_1.geom_type.__repr__())
                        adjacent_faces_graph.add_node(face_2, type=face_2.geom_type.__repr__())
                        adjacent_faces_graph.add_edge(
                            face_1,
                            face_2,
                            label=f_edge,
                        )
    # adjacent_faces_graph should have at least three connected subgraphs
    # (top side, bottom side, and sheet edge sides of the sheetmetal part).
    # We only care about the subgraph that includes the selected root face.
    nx.draw(adjacent_faces_graph, label="type", with_labels=True)
    plt.plot()
    for c in nx.connected_components(adjacent_faces_graph):
        if root_face in c:
            return adjacent_faces_graph.subgraph(c).copy()
    # If there is nothing tangent to the root face, return a graph with
    # one node and no edges.
    # This is useful for dxf/svg export of flat plates for manufacturing.
    single_face_graph = nx.Graph()
    single_face_graph.add_node(root_face)
    return single_face_graph    

class BendAllowanceCalculator:

    class KFactorStandard(Enum):
        ANSI = auto()
        DIN = auto()

    def __init__(self) -> None:
        self.k_factor_standard = None
        self.radius_thickness_values = None
        self.k_factor = None


    @classmethod
    def read_file(cls):
        instance = cls()

        radius_thickness_list = []
        k_factor_list = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "k-factor.csv")

        with open(file_path, mode="r", encoding="utf-8") as file: 
            reader = csv.reader(file)
            header = next(reader)
            a1 = header[0]
            b1 = header[1]
            r_t_header = "".join(c for c in a1 if c not in "' ").lower()
            if r_t_header != "radius/thickness":
                raise ValueError
            
            kf_header = "".join(c for c in b1 if c not in "' -()").lower()
            if kf_header == "kfactoransi":
                instance.k_factor_standard = cls.KFactorStandard.ANSI
            elif kf_header == "kfactordin":
                instance.k_factor_standard = cls.KFactorStandard.DIN
            else: 
                raise ValueError
            
            for row in reader: 
                if not row: 
                    continue
                radius_thickness_list.append(float(row[0]))
                k_factor_list.append(float(row[1]))

        instance.radius_thickness_values = radius_thickness_list
        instance.k_factor_values = k_factor_list

        return instance
    
    def get_k_factor(self, radius, thickness):
            r_over_t = radius / thickness
            if r_over_t <= self.radius_thickness_values[0]:
                kf_val = self.k_factor_values[0]
            elif r_over_t >= self.radius_thickness_values[-1]:
                kf_val = self.k_factor_values[-1]
            else:
                i = 0
                while r_over_t <= self.radius_thickness_values[i]:
                    i += 1
                kf1 = self.k_factor_values[i]
                kf2 = self.k_factor_values[i + 1]
                rt1 = self.radius_thickness_values[i]
                rt2 = self.radius_thickness_values[i + 1]
                kf_val = kf1 + (kf2 - kf1) * ((r_over_t - rt1) / (rt2 - rt1))
            return kf_val

    def get_bend_allowance(self, radius: float, thickness: float, bend_angle: float,
        ) -> float:
            factor = self.get_k_factor(radius, thickness)
            bend_allowance = (radius + factor * thickness) * bend_angle
            return bend_allowance 

class BendInfo:
    def __init__(self, parent_flange: Face, child_flange: Face, bend: Face, parent_bend_seam: Edge, child_bend_seam: Edge):
        self.parent_flange: Face = parent_flange
        self.child_flange: Face = child_flange
        self.bend: Face = bend
        self.parent_bend_seam: Edge = parent_bend_seam
        self.child_bend_seam: Edge = child_bend_seam
    
    def is_convex(self) -> bool:
        return self.bend.is_circular_convex
    
    def parent_local_plane(self) -> Plane:
        """Calculates the local plane to use as the local coordinate system during transforms

        Args:
            self (Bend): the bend which to calculate to local plane for 

        Raises:
            RuntimeError: Opencascade internal failures

        Returns:
            Solid: The resulting Solid object
        """
        local_origin = self.parent_bend_seam.start_point()  
        local_x = self.parent_bend_seam.tangent_at(0.5) 
        local_z = self.parent_flange.normal_at(local_origin)

        return Plane(
                origin=local_origin,
                x_dir=local_x,
                z_dir=local_z
            )

def _estimate_thickness(solid: Solid, reference_face: Face) -> float:
    bbox = reference_face.bounding_box()
    bbox_center = bbox.center()
    edge_x = bbox_center.X + (bbox.max.X - bbox.min.X) * 0.5
    edge = (edge_x, bbox_center.Y, bbox_center.Z)
    face_normal = reference_face.normal_at(edge)

    plan1 = Plane(origin=bbox_center, z_dir=face_normal)
    parallel_faces = solid.faces() | plan1

    opposite_faces = []
    for f in parallel_faces: 
        if f != reference_face:
           thickness =  f.distance_to(edge)
           opposite_faces.append([f, thickness])
    opposite_faces.sort(key=lambda x: x[1])
    thickness = opposite_faces[0][1]
    return thickness

def estimate_thickness(solid: Solid, ref_face: Face) -> float:
    # Get the normal of the reference face
    face_normal: Vector = ref_face.normal_at(
        ref_face.center()
    )

    # Find all faces parallel to the reference face
    parallel_faces: ShapeList[Face] = [
        f for f in solid.faces()
        if f.normal_at(f.center()).dot(face_normal)
    ]

    # Filter for the opposite face (anti-parallel normal)
    opposite_faces = [
        f for f in parallel_faces
        if (f.normal_at(f.center()).length - face_normal.length) < TOLERANCE and f.normal_at(f.center()).dot(face_normal) < 0
    ]

    if not opposite_faces:
        raise ValueError("No opposite face found for thickness estimation.")

    # Calculate the distance between the reference face and the opposite face
    thickness = ref_face.distance_to(opposite_faces[0])
    return thickness


""" def unbend_transforms(bend_sequence: List[Bend], thickness) -> List[Tuple[Face, Matrix]]: 
    import math
    from build123d import Spline, scale
    unfolded_flanges = []
    bockline = []

    from build123d.geometry import Rot, Location, Vector, Pos
    flange_transforms = {}
    if bend_sequence:
        # Why is this necessary?
        # Suggestions: initializing the transform for the first flange in the sequence with an empty Location
        flange_transforms[bend_sequence[0].parent_flange] = Location()
    


    for bend in bend_sequence:

        side1 = []
        side2 = []
         

        local_plane = bend.parent_local_plane() 

        to_local = Location(local_plane).inverse()
        to_world = Location(local_plane)

        local_p_flange = to_local * bend.parent_flange
        local_c_flange = to_local * bend.child_flange
        local_p_bend_seam = to_local * bend.parent_bend_seam
        local_c_bend_seam = to_local * bend.child_bend_seam
        n_parent = local_p_flange.normal_at()
        n_child = local_c_flange.normal_at()

        radie = bend.bend.radius
        cylinder_riktning = local_p_bend_seam.tangent_at(0.5)

        seam_center_point1 = local_p_bend_seam.center()
        seam_center_point2 = local_c_bend_seam.center()
        parent_bend_seam_y = seam_center_point1.Y
        child_bend_seam_y = seam_center_point2.Y

        new_x = seam_center_point2.X
        new_y = seam_center_point2.Y
        new_z = 0.0

        bend_angle_deg = n_parent.get_angle(n_child)
        if bend_angle_deg < 0.01:
            current_bend_location = Location()
        else:
            # longside are seam edges, both parent's and child's seam edges
            longside = []
            # 
            shortside_p = []
            # 
            shortside_c = []
            sides_cyl = [to_local * e for e in bend.bend.edges()]

            rotations_riktning = n_child.cross(n_parent)
            bend_angle_deg_ = bend_angle_deg * rotations_riktning

            p = seam_center_point1
            if abs(cylinder_riktning.X) > 0.999:
                flange_rotation = Pos(p.X, p.Y, 0) * Rot(bend_angle_deg_, 0, 0) * Pos(-p.X, -p.Y, 0)
                if parent_bend_seam_y > child_bend_seam_y: new_y += radie
                else: new_y -= radie

                for edge in sides_cyl:
                    show(edge)
                    if edge.length < 1.0: 
                        continue
                    start = edge.start_point()
                    end = edge.end_point()
                    middle = end - start

                    dist_start_p = local_p_flange.distance_to(start)
                    dist_end_p = local_p_flange.distance_to(end)
                    dist_start_c = local_c_flange.distance_to(start)
                    dist_end_c = local_c_flange.distance_to(end)

                    bbox = edge.bounding_box()
                    bbox_x = bbox.max.X - bbox.min.X
                    # Checks wether the edge is a curved edge in a funny way (not just a cylinder edge) - curved in the planar face plane

                    if (abs(middle.X) < 0.01 and abs(bbox_x) > 0.1) or (dist_start_p < 0.1 and dist_end_c < 0.1) or (dist_start_c < 0.1 and dist_end_p < 0.1):

                        plane1 = Plane(origin=local_p_bend_seam.center(), x_dir=cylinder_riktning, z_dir=n_parent)
                        plane2 = Plane(origin=local_c_bend_seam.center(), x_dir=cylinder_riktning, z_dir=n_child)
                        check = []
                        is_straight = True
                        previous_tangent = None
                        
                        for i in range(5):
                            t = i / 4
                            pt = edge.position_at(t)
                            local_pt = plane2.to_local_coords(pt)
                            flat_pt_local = Vector(local_pt.X, local_pt.Y, 0.0)
                            world_pt = plane2.from_local_coords(flat_pt_local)
                            if i > 0:
                                tangent = (world_pt - check[-1]).normalized()
                                if previous_tangent is not None:
                                    if tangent.cross(previous_tangent).length > 1e-5:
                                        is_straight = False
                                previous_tangent = tangent
                            check.append(world_pt)  

                        check_line = Spline(check)
                        if is_straight:
                            continue

                        distance1 = local_p_bend_seam.distance_to(edge.start_point())
                        distance2 = local_p_bend_seam.distance_to(edge.end_point())
                        if distance1 < 0.1: 
                            part1 = edge.trim(0.0, 0.5) 
                            part2 = edge.trim(0.5, 1.0)
                        elif distance2 < 0.1:
                            edge = edge.reversed()
                            part1 = edge.trim(0.0, 0.5) 
                            part2 = edge.trim(0.5, 1.0)
                        else: 
                            part1 = edge.trim(0.0, 0.5) 
                            part2 = edge.trim(0.5, 1.0)
                            dist1 = local_p_bend_seam.distance_to(part1.center())
                            dist2 = local_c_bend_seam.distance_to(part1.center())
                            if dist2 < dist1: 
                                part2 = edge.trim(0.0, 0.5) 
                                part1 = edge.trim(0.5, 1.0)

                        flat_points1 = []
                        flat_points2 = []
 
                        distance3 = abs(part1.start_point() - local_p_bend_seam.start_point())
                        distance4 = abs(part1.start_point() - local_p_bend_seam.end_point())
                        if distance3 < 5: 
                            flat_points1.append(local_p_bend_seam.start_point())
                        elif distance4 < 5: 
                            flat_points1.append(local_p_bend_seam.end_point())     

                        l = (edge.length)*2
                        num_points = math.ceil(l)  
                        if num_points < 2: num_points = 2
                        for i in range(num_points):
                            t = i / (num_points - 1)
                            pt1 = part1.position_at(t)
                            local_pt1 = plane1.to_local_coords(pt1)
                            flat_pt_local1 = Vector(local_pt1.X, local_pt1.Y, 0.0)
                            world_pt1 = plane1.from_local_coords(flat_pt_local1)
                            flat_points1.append(world_pt1)

                        for i in range(num_points):
                            t = i / (num_points - 1)
                            pt2 = part2.position_at(t)
                            local_pt2 = plane2.to_local_coords(pt2)
                            flat_pt_local2 = Vector(local_pt2.X, local_pt2.Y, 0.0)
                            world_pt2 = plane2.from_local_coords(flat_pt_local2)
                            flat_points2.append(world_pt2)   

                        distance5 = abs(part2.end_point() - local_c_bend_seam.start_point())
                        distance6 = abs(part2.end_point() - local_c_bend_seam.end_point())

                        if distance5 < 5: 
                            flat_points2.append(local_c_bend_seam.start_point())
                        elif distance6 < 5: 
                            flat_points2.append(local_c_bend_seam.end_point())   


                        cleaned_flat_points1 = []
                        cleaned_flat_points2 = []
                        for p in range(len(flat_points2)):
                            flat_point1 = flat_points1[p]
                            if flat_point1 not in cleaned_flat_points1:
                                cleaned_flat_points1.append(flat_point1)
                            flat_point2 = flat_points2[p]
                            if flat_point2 not in cleaned_flat_points2:
                                cleaned_flat_points2.append(flat_point2)

                        flat_edge1 = Spline(cleaned_flat_points1)
                        flat_edge2 = Spline(cleaned_flat_points2)
                        shortside_p.append(flat_edge1)
                        shortside_c.append(flat_edge2)
                        show(flat_edge1)
                        show(flat_edge2)
                   
                    elif (abs(bbox_x) > 0.01):                
                        if abs(edge.distance_to(local_p_bend_seam)) < abs(edge.distance_to(local_c_bend_seam)):
                            plane = Plane(origin=local_p_bend_seam.center(), x_dir=cylinder_riktning, z_dir=n_parent)
                        else: 
                            plane = Plane(origin=local_c_bend_seam.center(), x_dir=cylinder_riktning, z_dir=n_child)
                        flat_points = []

                        l = (edge.length)*2
                        num_points = math.ceil(l)  
                        if num_points < 2: num_points = 2
                        for i in range(num_points):
                            t = i / (num_points - 1)
                            pt = edge.position_at(t)

                            local_pt = plane.to_local_coords(pt)
                            flat_pt_local = Vector(local_pt.X, local_pt.Y, 0.0)
                            world_pt = plane.from_local_coords(flat_pt_local)
                            flat_points.append(world_pt)

                        flat_edge = Spline(flat_points)
                        longside.append(flat_edge)
                        show(flat_edge)
                longside = Wire.combine(longside)
                distance = longside[0].distance_to(local_p_bend_seam)
                if abs(distance) < 0.001: 
                    side1.append(longside[0])
                    side2.append(longside[1])
                else:
                    side2.append(longside[0])
                    side1.append(longside[1])

            new_child_center = Vector(new_x, new_y, new_z)
            current_child_center = local_c_bend_seam.center()
            relative_translation1 = new_child_center - current_child_center
            final_flange_location1 = Location(relative_translation1)

            current_bend_location = flange_rotation * final_flange_location1

            lista = BendAllowanceCalculator.read_file()
            import math
            bend_angle_r = bend_angle_deg* math.pi/180 
            bocken = lista.get_bend_allowence(radie, thickness, bend_angle_r)
            rikt = n_parent.cross(cylinder_riktning)
            förlängnings_vektor = rikt * bocken
            final_flange_location2 = Location(förlängnings_vektor)

            current_bend_location = final_flange_location2 * current_bend_location
            
            for side in shortside_p:
                side1.append(side)
            for side in shortside_c:
                side2.append(side)

            num_side1_wires = len(Wire.combine(side1))
            side1_combi = Wire.combine(side1)[0]
            side1_combi_edges = side1_combi.edges()

            num_side2_wires = len(Wire.combine(side2))
            side2_combi = Wire.combine(side2)[0]
            side2_combi_edges = side2_combi.edges()

            inside1 = []
            inside2 = []

            def edges_overlap(edge_a, edge_b, tolerance=1e-5):
                dist = (edge_a.center() - edge_b.center()).length
                return dist < tolerance 
                 
            if num_side1_wires != 1 and num_side2_wires != 1:
                for s in side1:
                    dist_p = local_p_bend_seam.distance_to(s @ 0)
                    dist_c = local_c_bend_seam.distance_to(s @ 0)
                    if dist_p < 0.01 or dist_c < 0.01:
                        continue
                    if not any(edges_overlap(s, combi_edge) for combi_edge in side1_combi_edges):
                        inside1.append(s)
                for s in side2:
                    dist_p = local_p_bend_seam.distance_to(s @ 0)
                    dist_c = local_c_bend_seam.distance_to(s @ 0)
                    if dist_p < 0.01 or dist_c < 0.01:
                        continue
                    if not any(edges_overlap(s, combi_edge) for combi_edge in side2_combi_edges):
                        inside2.append(s)
            
            inside1_compound = Compound(inside1)
            inside2_compound = Compound(inside2) 
        
        parent_transform = flange_transforms.get(parent_flange, Location())
        global_current_bend_location = to_world * current_bend_location * to_local
        child_transform = parent_transform * global_current_bend_location
        flange_transforms[child_flange] = child_transform

        test_utbredd_flans = child_transform * child_flange
        unfolded_flanges.append(test_utbredd_flans)
        if bend_angle_deg > 0.01:
            dir_p_start = (side1_combi % 0)
            dir_c_start = (side2_combi % 0)
            dir_p_end = (side1_combi % 1)
            dir_c_end = (side2_combi % 1)

            dot_a = abs(dir_p_start.dot(cylinder_riktning))
            dot_b = abs(dir_c_start.dot(cylinder_riktning))
            dot_c = abs(dir_p_end.dot(cylinder_riktning))
            dot_d = abs(dir_c_end.dot(cylinder_riktning))

            parent_bend_seam_transformed = parent_transform * to_world * side1_combi
            child_bend_seam_transformed = child_transform * to_world * side2_combi

            inside_p_seam = parent_transform * to_world * inside1_compound
            inside_c_seam = child_transform * to_world * inside2_compound

            if abs(dot_a - 1) > 0.01 or abs(dot_b - 1) > 0.01 or abs(dot_c - 1) > 0.01 or abs(dot_d - 1) > 0.01:
                point_a_c, point_b_c = parent_flange.closest_points(test_utbredd_flans)
                dist_vec_c = point_a_c - point_b_c
                norm_dist_vec_c = dist_vec_c.normalized()

                point_a_p, point_b_p = test_utbredd_flans.closest_points(parent_flange)
                dist_vec_p = point_a_p - point_b_p
                norm_dist_vec_p = dist_vec_p.normalized()

                center_p = parent_bend_seam_transformed.center()
                center_c = child_bend_seam_transformed.center()

                scaling_plane_p = Plane(origin=center_p, z_dir=norm_dist_vec_p)
                scaling_plane_c = Plane(origin=center_c, z_dir=norm_dist_vec_c)

                scaling_plane_p_to_local = Location(scaling_plane_p).inverse()
                scaling_plane_p_to_world = Location(scaling_plane_p)

                scaling_plane_c_to_local = Location(scaling_plane_c).inverse()
                scaling_plane_c_to_world = Location(scaling_plane_c)

                local_side_p = scaling_plane_p_to_local * parent_bend_seam_transformed
                local_side_c = scaling_plane_c_to_local * child_bend_seam_transformed
                bbox_p = local_side_p.bounding_box()
                bbox_c = local_side_c.bounding_box()

                height_p_local = max(bbox_p.max.Z - bbox_p.min.Z, 1e-6)
                height_c_local = max(bbox_c.max.Z - bbox_c.min.Z, 1e-6)

                scale_factor_p_z = 1.0 if dist_vec_p.length < 0.1 else (dist_vec_p.length * 0.5) / height_p_local
                scale_factor_c_z = 1.0 if dist_vec_c.length < 0.1 else (dist_vec_c.length * 0.5) / height_c_local

                strech_side1_local = scale(local_side_p, by=(1.0, 1.0, scale_factor_p_z), about=(0, 0, 0))
                strech_side2_local = scale(local_side_c, by=(1.0, 1.0, scale_factor_c_z), about=(0, 0, 0))

                parent_bend_seam_transformed = scaling_plane_p_to_world * strech_side1_local
                child_bend_seam_transformed = scaling_plane_c_to_world * strech_side2_local

            inside = []

            if inside_p_seam:
                for h in range(len(inside_p_seam.edges())): 
                    h1 = inside_p_seam.edges()[h]
                    h2 = Edge.make_line(h1 @ 1, inside_c_seam.edges()[h].start_point())
                    h3 = inside_c_seam.edges()[h]
                    h4 = Edge.make_line(h3 @ 1, h1 @ 0)
                    h = h1 + h2 + h3 + h4
                    inside.append(h)

            l1 = parent_bend_seam_transformed
            l3 = child_bend_seam_transformed 

            gap_1_3 = parent_bend_seam_transformed.distance_to(child_bend_seam_transformed)
            gap_3_1 = child_bend_seam_transformed.distance_to(parent_bend_seam_transformed)

            if gap_1_3 == 0 and gap_3_1 == 0: 
                l = l1 + l3
                bock_point1 = parent_bend_seam_transformed @ 0
                bock_point2 = parent_bend_seam_transformed @ 1
            else: 
                l2 = Edge.make_line(l1 @ 1, child_bend_seam_transformed @ 0)
                l4 = Edge.make_line(l3 @ 1, l1 @ 0)
                bock_point1 = l2.center()
                bock_point2 = l4.center()
                l = l1 + l2 + l3 + l4

            sheet_fold = Face(l)
            if inside:
                for i in range(len(inside)):
                    edge = inside[i]
                    hole = Face(edge)
                    sheet_fold -= hole
                    
            bockline.append([bock_point1, bock_point2])

            unfolded_flanges.append(sheet_fold)

    return unfolded_flanges, bockline """



