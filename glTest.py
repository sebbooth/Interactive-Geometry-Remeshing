import pygame
import numpy as np
import igl
import math
import meshplot as mp
import scipy as sp
import ipywidgets as iw

from PIL import Image
from PIL import ImageOps

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def feature_edges(v,f,threshold):
    edge_faces = []

    dihedral_threshold = threshold/180 * np.pi
    f_normals = igl.per_face_normals(v,f,np.ndarray([0,0]))
    edge_flaps = igl.edge_flaps(f)
    edges = edge_flaps[0]
    
    threshold_edges = []
    for i in range(edges.shape[0]):
        # find normals of faces connecting edge
        edge_face_norms = f_normals[edge_flaps[2][i]]

        # find dihedral angle between edge faces
        f1n = edge_face_norms[0] / np.linalg.norm(edge_face_norms[0])
        f2n = edge_face_norms[1] / np.linalg.norm(edge_face_norms[1])
        dihedral_angle = np.arccos(np.clip(np.dot(f1n, f2n), -1.0, 1.0))

        # add angles greater than threshold to list
        if dihedral_angle > dihedral_threshold or (-1 in edge_flaps[2][i]):
            threshold_edges.append(edges[i])
            
            # add edge faces to list
            if edge_flaps[2][i][0] != -1:
                edge_faces.append(edge_flaps[2][i][0])
            if edge_flaps[2][i][1] != -1:
                edge_faces.append(edge_flaps[2][i][1])

    threshold_edges = np.array(threshold_edges)
    return threshold_edges, edge_faces

def circle_to_square(x_circ,y_circ):
    x = 0.5 * np.sqrt( 2 + x_circ**2 - y_circ**2 + 2*x_circ*np.sqrt(2) ) - 0.5 * np.sqrt( 2 + x_circ**2 - y_circ**2 - 2*x_circ*np.sqrt(2) )
    y = 0.5 * np.sqrt( 2 - x_circ**2 + y_circ**2 + 2*y_circ*np.sqrt(2) ) - 0.5 * np.sqrt( 2 - x_circ**2 + y_circ**2 - 2*y_circ*np.sqrt(2) )
    x = max(min(1,x),-1)
    y = max(min(1,y),-1)
    return [x,y]

def parametrize(v,f):
    ## Find the open boundary
    bnd = igl.boundary_loop(f)
    
    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(v, bnd)
    
    # remap circle boundary onto square
    bnd_uv = np.array([circle_to_square(x_circ,y_circ) for [x_circ,y_circ] in bnd_uv])

    ## Harmonic parametrization for the internal vertices
    return igl.harmonic(v, f, bnd, bnd_uv, 1)


v, f = igl.read_triangle_mesh("data/fandisk.obj")
uv = parametrize(v,f)
v_p = np.hstack([uv, np.zeros((uv.shape[0],1))])
vertices = list(v_p)
faces = list(f)

def Render():
    glBegin(GL_TRIANGLES)
    for face in faces:
        for vertex in face:
            glVertex3fv(tuple(vertices[vertex]))
    glEnd()

def main():
    pygame.init()
    width = 1000
    height = 1000
    display = (width,height)
    window = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    glOrtho(-1.0,1.0,-1.0,1.0,1.0,-1.0)

    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:               
                pygame.quit()
                quit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Render()
        pygame.display.flip()
        pygame.time.wait(10)

main()