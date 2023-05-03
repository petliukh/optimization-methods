from __future__ import annotations
import numpy as np
import scipy as sp
import pandas as pd
from enum import Enum, auto
import math


Matrix = list[list[float]]


class ProblemType(Enum):
    MIN = auto()
    MAX = auto()


class GaussianJordanElimination:
    def __init__(self, matrix: Matrix | np.ndarray):
        self.m = np.array(matrix)
        self.basis_cols = np.array([], dtype=np.int32)

    def eliminate(self, eps=1e-4):
        rows, _ = self.m.shape
        i = 0
        s = 0
        while i < rows:
            if math.isclose(self.m[i][s], 0, abs_tol=eps):
                for row in range(i + 1, rows):
                    if not math.isclose(self.m[row][s], 0, abs_tol=eps):
                        self.m[[i, row]] = self.m[[row, i]]
                        break
                else:
                    s += 1
                    continue
            if not math.isclose(self.m[i][s], 1, abs_tol=eps):
                self.m[i] /= self.m[i][s]
            for j in range(rows):
                if i == j:
                    continue
                if math.isclose(self.m[i][s], 0, abs_tol=eps):
                    continue
                self.m[j] = -self.m[j][s] * self.m[i] + self.m[j]
            self.basis_cols = np.append(self.basis_cols, s)
            i += 1
            s += 1
        return self.m


class SimplexMinMax:
    def __init__(
        self,
        system: Matrix | np.ndarray,
        signs: list[str],
        r_vec: list[float],
        tar_func: list[float],
        var_constraints: list[tuple[int, str]] | None = None,
        problem_type: ProblemType = ProblemType.MIN,
    ) -> None:
        assert len(signs) == len(
            system
        ), "Signs length should be equal to the number of rows in the system."
        self.m = np.array(system, dtype=np.float64)
        self.signs = signs
        self.r_vec = np.array(r_vec, dtype=np.float64)
        self.slack_vars = []
        self.basis_cols = np.array([], dtype=np.int32)
        self.tar_func = np.array(tar_func)
        self.var_constraints = var_constraints or []
        self.problem_type = problem_type
        self.var_name = "x"
        self.slack_name = "s"

    def add_slack_var(self, row):
        self.slack_vars.append(row)
        self.m = np.c_[self.m, np.zeros(len(self.m)).T]
        self.m[row][-1] = 1
        self.tar_func = np.append(self.tar_func, 0)

    def make_correct_signs(self):
        for i, sign in enumerate(self.signs):
            if self.problem_type == ProblemType.MIN:
                if sign == "<=":
                    self.m[i] = -self.m[i]
                    self.r_vec[i] = -self.r_vec[i]
                    self.signs[i] = ">="
            elif self.problem_type == ProblemType.MAX:
                if sign == ">=":
                    self.m[i] = -self.m[i]
                    self.r_vec[i] = -self.r_vec[i]
                    self.signs[i] = "<="
            else:
                raise ValueError("Incorrect problem type.")

    def make_equalities(self):
        for i, sign in enumerate(self.signs):
            if sign != "=":
                self.add_slack_var(i)

    def make_constant_vec_positive(self):
        for i, val in enumerate(self.r_vec):
            if val < 0:
                self.r_vec[i] *= -1
                self.m[i] *= -1

    def make_unit_basis_cols(self):
        full_mtx = np.c_[self.m, self.r_vec.T]
        gje = GaussianJordanElimination(full_mtx)
        gje.eliminate()
        cols = self.m.shape[1]
        self.m = gje.m[:, :cols]
        self.r_vec = gje.m[:, cols].ravel()
        self.basis_cols = gje.basis_cols

    def reduce_to_canonical_form(self):
        self.make_correct_signs()
        self.make_equalities()
        self.make_constant_vec_positive()
        self.make_unit_basis_cols()

    def calc_output_vec(self):
        output_vec = []
        full_mtx = np.c_[self.m, self.r_vec.T]
        basis_coeffs = self.tar_func[self.basis_cols]
        for col in range(full_mtx.shape[1]):
            output_vec.append(np.dot(basis_coeffs, full_mtx[:, col]))
        return self.tar_func - output_vec[:-1], output_vec[-1]

    def find_pivots(self, output_vec):
        full_mtx = np.c_[self.m, self.r_vec.T]
        max = output_vec[0]
        pivot_col = 0

        for i, val in enumerate(output_vec):
            if val > max and i not in self.basis_cols:
                max = val
                pivot_col = i

        ratios = []
        for i in range(full_mtx.shape[0]):
            if not any(x > 0 for x in full_mtx[:, pivot_col].ravel()):
                raise ValueError(f"Optimal solution does not exist!")
            if full_mtx[i][pivot_col] > 0:
                ratios.append((i, self.r_vec[i] / full_mtx[i][pivot_col]))

        pivot_row = min(ratios, key=lambda x: x[1])[0]
        pivot_elem = full_mtx[pivot_row][pivot_col]

        return pivot_row, pivot_col, pivot_elem

    def change_basis(self, output_vec):
        pivot_row, pivot_col, pivot_elem = self.find_pivots(output_vec)
        full_mtx = np.c_[self.m, self.r_vec.T]
        full_mtx[pivot_row] /= pivot_elem
        cols = self.m.shape[1]

        for i in range(full_mtx.shape[0]):
            if i == pivot_row:
                continue
            mult = full_mtx[i][pivot_col]
            full_mtx[i] = -mult * full_mtx[pivot_row] + full_mtx[i]

        self.m = full_mtx[:, :cols]
        self.r_vec = full_mtx[:, cols].ravel()
        self.basis_cols[self.basis_cols == pivot_row] = pivot_col

    def print_iter(self, output_vec, z, num=None):
        def round_and_str(val):
            return str(round(val, 2))

        base_vars = self.m.shape[0]

        index = [
            f"{self.var_name if i < base_vars else self.slack_name}_{i if i < base_vars else i - base_vars}"
            for i in self.basis_cols
        ]
        full_mtx = np.c_[self.m, self.r_vec]

        header = [
            f"{self.var_name if i < base_vars else self.slack_name}_{i if i < base_vars else i - base_vars}"
            for i in range(self.m.shape[1])
        ]
        header.insert(0, "C_b")
        header.append("b")
        indent = full_mtx.shape[0] // 2 + 1

        print("\t" * indent + f"---=== Iteration {num} ===---")

        print("\t" + "\t".join(header))
        print("\t\t" + "\t".join(map(round_and_str, self.tar_func)))

        c_b = np.array([self.tar_func[i] for i in self.basis_cols])

        for i, row in enumerate(full_mtx):
            print(f"{index[i]}\t" + f"{c_b[i]}\t" + "\t".join(map(round_and_str, row)))

        print("\tz_j\t", "\t".join(map(round_and_str, output_vec)) + f"\t{z:.2f}")

    def print_problem(self):
        def round_and_str(val):
            return str(round(val, 2))

        if self.problem_type == ProblemType.MIN:
            print("Minimization problem:")
        elif self.problem_type == ProblemType.MAX:
            print("Maximization problem:")
        else:
            raise ValueError("Incorrect problem_type")

        print("C_b = [", end="")
        print(", ".join(map(round_and_str, self.tar_func)), end="")
        print("]")
        print("\nA = ")
        print(self.m, "\n")
        print(self.signs)
        print("\nb = ")
        print(self.r_vec, end="\n\n")
        print(self.var_constraints, end="\n\n")

    def get_opposite_problem_type(self):
        return (
            ProblemType.MAX if self.problem_type == ProblemType.MIN else ProblemType.MIN
        )

    def make_dual(self):
        self.make_correct_signs()

        old_var_constr = self.var_constraints
        old_signs = self.signs

        self.tar_func, self.r_vec = self.r_vec, self.tar_func
        self.m = self.m.T
        self.problem_type = self.get_opposite_problem_type()

        self.var_constraints = []
        self.signs = ["="] * len(self.signs)

        for i, s in old_var_constr:
            if s != "=":
                if self.problem_type == ProblemType.MIN:
                    self.signs[i] = ">="
                elif self.problem_type == ProblemType.MAX:
                    self.signs[i] = "<="
                else:
                    raise ValueError("Incorrect problem type.")

        for i, s in enumerate(old_signs):
            if s != "=":
                self.var_constraints.append((i, ">="))

        self.var_name = "y" if self.var_name == "x" else "x"

    def solve(self, eps, print_iter=False):
        if print_iter:
            self.print_problem()
        self.reduce_to_canonical_form()

        def optimal(output_vec, eps):
            return all(i < 0 or math.isclose(i, 0, abs_tol=eps) for i in output_vec)

        output_vec, z = self.calc_output_vec()

        if print_iter:
            self.print_iter(output_vec, z, num=1)

        i = 2
        while not optimal(output_vec, eps):
            self.change_basis(output_vec)
            output_vec, z = self.calc_output_vec()
            if print_iter:
                self.print_iter(output_vec, z, num=i)
            i += 1

        sol = [output_vec[i] if i in self.basis_cols else 0 for i in range(len(self.m))]

        if print_iter:
            print(f"\nX_min = {sol}, F = {z}")

        return sol, z


def test_deco(func):
    def wrapper(*args, **kwargs):
        print(f"---=== Running test '{func.__name__}' ===---")
        try:
            func(*args, **kwargs)
        except (AssertionError, Exception) as ex:
            print(f"\nERROR --- TEST FAILED: {ex}\n")
            print("=" * 80)
            return
        print(f"\n OK --- TEST PASSED SUCCESSFULLY\n")
        print("=" * 80)

    return wrapper


@test_deco
def test_reduce_to_canonical_form():
    tar_func = [-1.0, 5, 1, -1]
    system = [
        [1.0, 3, 3, 1],
        [2, 0, 3, -1],
    ]
    signs = ["=", ">="]
    r_vec = [3.0, 4]
    smpx = SimplexMinMax(system, signs, r_vec, tar_func)
    smpx.reduce_to_canonical_form()

    assert len(smpx.slack_vars) == 1
    assert smpx.slack_vars[0] == 1
    assert np.all(smpx.r_vec > 0)
    assert 2 == len(smpx.basis_cols)


@test_deco
def test_gaussian_jordan_elim():
    system = [
        [1.0, 0, 0, 1],
        [2, 0, 3, -1],
    ]
    eps = 1e-3
    gje = GaussianJordanElimination(system)
    gje.eliminate()
    basis_cols = []

    for col in range(gje.m.shape[1]):
        if math.isclose(np.sum(gje.m[:, col]), 1, abs_tol=eps):
            basis_cols.append(col)

    assert 2 <= len(basis_cols)


@test_deco
def test_simplex_method_solve():
    tar_func = [-1.0, 5, 1, -1]
    system = [
        [1.0, 3, 3, 1],
        [2, 0, 3, -1],
    ]
    signs = ["=", ">="]
    r_vec = [3.0, 4]
    smpx = SimplexMinMax(system, signs, r_vec, tar_func)
    solution_vec, z = smpx.solve(1e-3, print_iter=True)

    assert solution_vec.shape[0] == 4


@test_deco
def test_simplex_method_solve_2():
    tar_func = [-7.0, 6, 4]
    system = [
        [-3.0, 1, 0],
        [-4, 2, 0],
        [-1, 1, 1],
    ]
    signs = ["<=", "<=", "<="]
    r_vec = [-1.0, -4, -1]
    smpx = SimplexMinMax(system, signs, r_vec, tar_func)
    solution_vec, z = smpx.solve(1e-3, print_iter=True)

    assert solution_vec.shape[0] == 3


@test_deco
def test_simplex_method_solve_dual():
    tar_func = [1.0, -4, -1]
    system = [
        [3.0, 4, 1],
        [1, 2, 1],
        [0, 0, 1],
    ]
    signs = ["<=", "=", ">="]
    r_vec = [7.0, 6, 4]
    var_constraints = [
        (0, ">="),
        (1, ">="),
        (2, ">="),
    ]
    smpx = SimplexMinMax(system, signs, r_vec, tar_func, var_constraints)
    smpx.print_problem()
    smpx.make_dual()
    solution_vec, z = smpx.solve(1e-3, print_iter=True)

    assert len(solution_vec) == 3


def run_tests():
    # test_reduce_to_canonical_form()
    # test_gaussian_jordan_elim()
    # test_simplex_method_solve()
    # test_simplex_method_solve_2()
    test_simplex_method_solve_dual()


if __name__ == "__main__":
    run_tests()
