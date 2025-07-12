import numpy as np
import plotly.graph_objects as go
from manotorch.manolayer import MANOOutput, ManoLayer


def _mesh_plot(vertices, faces, color=None, opacity=.5):
    if vertices.shape[0] != 3 or faces.shape[0] != 3:
        vertices = vertices.T
        faces = faces.T

    mesh3d_kwargs = dict(
        x=vertices[0],
        y=vertices[1],
        z=vertices[2],
        i=faces[0],
        j=faces[1],
        k=faces[2],
        opacity=opacity)

    if color is not None:
        mesh3d_kwargs.update(color=color)

    return go.Mesh3d(**mesh3d_kwargs)


def mesh_plot(mesh, color=None, opacity=.5):
    return _mesh_plot(mesh.vertices, mesh.faces, color, opacity)


def mano_plot(mano_output: MANOOutput, mano_layer: ManoLayer, closed_faces=True, opacity=.5):
    faces = mano_layer.get_mano_closed_faces() if closed_faces else mano_layer.th_faces
    return _mesh_plot(mano_output.verts[0], faces)


def scatter_plot(vertices, colors=None, opacity=.5, colorscale='Bluered'):
    if isinstance(vertices, list):
        vertices = np.array(vertices)
    if vertices.size == 3:
        vertices = vertices.reshape(3, -1)
    if vertices.shape[1] == 3:
        vertices = vertices.T

    scatter_3d_kwargs = dict(
        x=vertices[0],
        y=vertices[1],
        z=vertices[2],
        mode='markers')

    if colors is not None:
        marker_dict = scatter_3d_kwargs.get('marker', dict())
        marker_dict.update(
            color=colors.flatten(),
            opacity=opacity)
        scatter_3d_kwargs['marker'] = marker_dict

    if colorscale:
        marker_dict = scatter_3d_kwargs.get('marker', dict())
        marker_dict['colorscale'] = colorscale
        scatter_3d_kwargs['marker'] = marker_dict

    return go.Scatter3d(**scatter_3d_kwargs)


def line_plot(from_point, to_point, colors=None, opacity=.5, colorscale='Bluered'):
    vertices = np.vstack([from_point, to_point]).T

    scatter_3d_kwargs = dict(
        x=vertices[0],
        y=vertices[1],
        z=vertices[2],
        mode='lines')

    if colors is not None:
        marker_dict = scatter_3d_kwargs.get('marker', dict())
        marker_dict.update(
            color=colors.flatten(),
            colorscale=colorscale,
            opacity=opacity)
        scatter_3d_kwargs['marker'] = marker_dict

    return go.Scatter3d(**scatter_3d_kwargs)


def armature_plot(vertices, edge_indexes):
    if vertices.shape[0] != 3:
        vertices = vertices.T
        edge_indexes = edge_indexes.T

    # take the coordinates of the edges
    edges = vertices.take(edge_indexes.ravel(), axis=1)

    # insert a None between pairs of coordinates that define an edge
    # plotly interprets this as the end of a line
    between_pairs = range(2, edges.shape[-1] + 1, 2)
    # noinspection PyTypeChecker
    edges = np.insert(
        edges,
        between_pairs,
        None,
        axis=-1)

    return go.Scatter3d(
        x=edges[0],
        y=edges[1],
        z=edges[2],
        mode='lines+markers')


def figure(*plots, title=None):
    fig = go.Figure(
        plots,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor='rgba(0, 0, 0,0)',
                    gridcolor='white',
                    showbackground=False,
                    zerolinecolor='white', ),

                yaxis=dict(
                    backgroundcolor='rgba(0, 0, 0,0)',
                    gridcolor='white',
                    showbackground=False,
                    zerolinecolor='white'),

                zaxis=dict(
                    backgroundcolor='rgba(0, 0, 0,0)',
                    gridcolor='white',
                    showbackground=False,
                    zerolinecolor='white'),
            )))
    if title:
        fig.update_layout(title=dict(text=title))
    fig.update_layout(scene_aspectmode='data')
    return fig


def animation(components, title=None, titles=None):
    figures = [go.Frame(data=component) for component in components]

    play_button = dict(
        label='Play',
        method='animate',
        args=[None])

    menu = dict(
        type='buttons',
        buttons=[play_button])

    fig = go.Figure(
        frames=figures,
        data=figures[0].data,
        layout=go.Layout(
            scene=dict(
                aspectmode='cube',
                aspectratio=dict(x=1, y=1, z=1)
            ),
            updatemenus=[menu]))

    if title:
        fig.update_layout(title=dict(text=title))
    if titles:
        for i in range(len(fig.frames)):
            fig.frames[i]['layout'].update(title_text=titles[i])
    fig.update_layout(transition_duration=5000)
    fig.update_layout(scene_aspectmode='data')
    return fig
