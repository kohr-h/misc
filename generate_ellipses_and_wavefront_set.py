import numpy as np
import odl


max_ab = 0.5
space = space = odl.uniform_discr([-1, -1], [1, 1], (128, 128))
pts = space.points()


def ellipse(fval, center, a, b, phi):
    test_pts = pts - np.array(center)[None, :]
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)],
                        [np.sin(phi), np.cos(phi)]])
    test_pts = rot_mat.T.dot(test_pts.T).T
    inside = np.where(
        (test_pts[:, 0] / a) ** 2 + (test_pts[:, 1] / b) ** 2 <= 1)[0]
    ell = np.zeros(space.shape)
    ell.ravel()[inside] = fval
    return ell


def ellipse_edge_angle(center, a, b, phi):
    test_pts = pts - np.array(center)[None, :]
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)],
                        [np.sin(phi), np.cos(phi)]])
    test_pts = rot_mat.T.dot(test_pts.T).T
    eps = 4 * np.max(space.cell_sides) / np.min([a, b])
    cond_pts = (test_pts[:, 0] / a) ** 2 + (test_pts[:, 1] / b) ** 2
    bdry = np.where((cond_pts >= 1 - eps) & (cond_pts <= 1 + eps))[0]

    normal = ((test_pts[:, 0] / a) / np.sqrt(cond_pts),
              (test_pts[:, 1] / b) / np.sqrt(cond_pts))

    angle = np.zeros(space.shape)
    angle.ravel()[bdry] = np.arctan2(normal[1][bdry], normal[0][bdry])
    return np.mod(angle, np.pi)


grad = odl.Gradient(space, pad_mode='symmetric')
pwnorm = odl.PointwiseNorm(grad.range)


def random_ellipses(n):
    assert n >= 1

    fval = np.random.uniform(0.1, 2)
    center = np.random.uniform(-1, 1, size=2)
    a, b = np.random.uniform(0.05, max_ab, size=2)
    phi = np.random.uniform(0, np.pi)

    ell = ellipse(fval, center, a, b, phi)

    grad_ell = [x.asarray() for x in grad(ell)]
    grad_ell_norm = pwnorm(grad_ell).asarray()
    edge = grad_ell_norm > fval / (2 * space.cell_sides[0])

    edge_angles = ellipse_edge_angle(center, a, b, phi)
    edge_angles[edge == 0] = 0

    for _ in range(n - 1):
        fval = np.random.uniform(0.1, 2)
        center = np.random.uniform(-1, 1, size=2)
        a, b = np.random.uniform(0.05, max_ab, size=2)
        phi = np.random.uniform(0, np.pi)

        ell0 = ellipse(fval, center, a, b, phi)
        ell += ell0

        grad_ell0 = [x.asarray() for x in grad(ell0)]
        grad_ell0_norm = pwnorm(grad_ell0).asarray()

        edge0 = grad_ell0_norm > fval / (2 * space.cell_sides[0])
        edge[:] = edge.astype(bool) | edge0.astype(bool)

        edge_angles0 = ellipse_edge_angle(center, a, b, phi)
        edge_angles0[edge0 == 0] = 0
        single = (edge0 != 0) & (edge_angles == 0)
        double = (edge0 != 0) & (edge_angles != 0)
        edge_angles[single] = edge_angles0[single]
        edge_angles[double] = (edge_angles[double] + edge_angles0[double]) / 2

    return tuple(space.element(x) for x in (ell, edge, edge_angles))


ell, edge, edge_angles = random_ellipses(20)
ell.show('Ellipses', force_show=True)
edge.show('Edges', force_show=True)
edge_angles.show('Edge angles', force_show=True)
