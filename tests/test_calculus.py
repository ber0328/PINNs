import pytest
import torch
import sys
sys.path.append('../')
from src import calculus as calc


@pytest.fixture
def scalar_func():
    return lambda x: (x[:, 0:1]**2 + x[:, 1:2]**2) * x[:, 2:3]


@pytest.fixture
def vec_func():
    return lambda x: torch.cat([torch.sin(x[:, 0:1]) + x[:, 1:2]**2,
                                torch.cos(x[:, 2:3] + x[:, 1:2])], dim=1)


@pytest.fixture
def vec_func_2():
    return lambda x: torch.cat([torch.sin(x[:, 0:1]) + x[:, 1:2],
                                torch.cos(x[:, 1:2]) + x[:, 0:1]], dim=1)


@pytest.fixture
def vec_func_2_convective_der():
    return lambda x: torch.cat([torch.cos(x[:, 0:1]) * (torch.sin(x[:, 0:1]) + x[:, 1:2]) + torch.cos(x[:, 1:2]) + x[:, 0:1],
                                torch.sin(x[:, 0:1]) + x[:, 1:2] - torch.sin(x[:, 1:2]) * (torch.cos(x[:, 1:2]) + x[:, 0:1])], dim=1)


@pytest.fixture
def vec_func_2_laplacian():
    return lambda x: torch.cat([-torch.sin(x[:, 0:1]),
                                -torch.cos(x[:, 1:2])], dim=1)


@pytest.fixture
def vec_func_mat_der():
    return lambda x: torch.cat([torch.cos(x[:, 0:1]) * (torch.sin(x[:, 0:1]) + x[:, 1:2]**2) + 2 * x[:, 1:2] * torch.cos(x[:, 2:3] + x[:, 1:2]),
                                - 0.5 * torch.sin(2 * (x[:, 2:3] + x[:, 1:2])) - torch.sin(x[:, 1:2] + x[:, 2:3])], dim=1)


@pytest.fixture
def vec_func_div():
    return lambda x: torch.cos(x[:, 0:1]) - torch.sin(x[:, 1:2] + x[:, 2:3])


@pytest.fixture
def vec_func_laplacian():
    return lambda x: torch.cat([2 - torch.sin(x[:, 0:1]), -torch.cos(x[:, 1:2] + x[:, 2:3])], dim=1)


@pytest.fixture
def scalar_func_grad():
    return lambda x: torch.cat([2 * x[:, 0:1] * x[:, 2:3], 2 * x[:, 1:2] * x[:, 2:3]], dim=1)


@pytest.fixture
def scalar_func_laplacian_time_indep():
    return lambda x: 4 * x[:, 2:3]


@pytest.fixture
def input_data():
    return torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [0, 0, 0]], requires_grad=True)


@pytest.fixture
def input_data_2d():
    return torch.tensor([[1.0, 1.0], [1.0, 2.0], [0, 0]], requires_grad=True)


def test_material_derivative(input_data, vec_func, vec_func_mat_der):
    output_data = vec_func(input_data)
    assert torch.allclose(calc.material_derivative(input_data, output_data, time_dependant=True), vec_func_mat_der(input_data))


def test_div(input_data, vec_func, vec_func_div):
    output_data = vec_func(input_data)
    assert torch.allclose(calc.div(input_data, output_data, True), vec_func_div(input_data))


def test_laplacian(input_data, vec_func, vec_func_laplacian):
    output_data = vec_func(input_data)
    assert torch.allclose(calc.laplacian(input_data, output_data), vec_func_laplacian(input_data))


def test_dir_derivative(input_data, scalar_func, scalar_func_grad):
    output_data = scalar_func(input_data)
    direction = torch.tensor([1.0, 0.0]).unsqueeze(1)
    result = calc.dir_derivative(input_data, output_data, direction)
    expected_result = scalar_func_grad(input_data) @ direction
    assert torch.allclose(result, expected_result)


def test_scalar_laplacian(input_data, scalar_func, scalar_func_laplacian_time_indep):
    output_data = scalar_func(input_data)
    laplacian_result = calc.laplacian(input_data, output_data, time_dependant=False)
    expected_laplacian = scalar_func_laplacian_time_indep(input_data)
    assert torch.allclose(laplacian_result, expected_laplacian)


def test_convective_derivative(input_data_2d, vec_func_2, vec_func_2_convective_der):
    output_data = vec_func_2(input_data_2d)
    result = calc.material_derivative(input_data_2d, output_data, time_dependant=False)
    expected_result = vec_func_2_convective_der(input_data_2d)
    print(result)
    print(expected_result)
    assert torch.allclose(result, expected_result)


def test_vec_2_laplacian(input_data_2d, vec_func_2, vec_func_2_laplacian):
    output_data = vec_func_2(input_data_2d)
    result = calc.laplacian(input_data_2d, output_data, time_dependant=False)
    expected_result = vec_func_2_laplacian(input_data_2d)
    assert torch.allclose(result, expected_result)


def test_nabla(input_data, scalar_func, scalar_func_grad):
    output_data = scalar_func(input_data)
    result = calc.nabla(input_data, output_data)  # Exclude time dimension
    expected_result = scalar_func_grad(input_data)
    assert torch.allclose(result, expected_result)


@pytest.fixture
def int_functions():
    return (lambda x: x[:, 0:1]**2 + x[:, 1:2]**2, lambda x: x[:, 0:1] + x[:, 1:2])


def test_L2_norm(int_functions):
    f, g = int_functions
    norm = calc.L2_norm(f, g, 2, [1, 1], [-1, -1])
    print(norm)
    assert torch.allclose(norm, torch.tensor([2.27058484879]))
