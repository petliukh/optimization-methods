import numpy as np
import io


def exchange_diff(a, b):
    if a > b:
        return a - b, 0
    return 0, b - a


class FogelMethod:
    def __init__(
        self,
        tariffs,
        demand,
        supply,
    ) -> None:
        self.tariffs = np.array(tariffs, dtype=np.float64)
        self.prod_qnts = np.zeros((len(supply), len(demand)))
        self.supply = np.array(supply, dtype=np.float64)
        self.demand = np.array(demand, dtype=np.float64)

        self.row_penalties = []
        self.col_penalties = []

        self.skip_rows = []
        self.skip_cols = []

    def find_row_penalties(self):
        row_penalties = []
        tariffs_mask = self.get_tariffs_mask()

        for i, row in enumerate(tariffs_mask):
            if i in self.skip_rows:
                row_penalties.append(np.nan)
                continue

            row[np.isnan(row)] = np.inf
            min1, min2 = np.sort(row)[:2]
            row_penalties.append(np.abs(min1 - min2))

        row_penalties = np.array(row_penalties)
        row_penalties = np.ma.array(row_penalties, mask=np.isnan(row_penalties))
        return row_penalties

    def find_column_penalties(self):
        col_penalties = []
        tariffs_mask = self.get_tariffs_mask()
        for j, col in enumerate(tariffs_mask.T):
            if j in self.skip_cols:
                col_penalties.append(np.nan)
                continue

            col[np.isnan(col)] = np.inf
            min1, min2 = np.sort(col)[:2]
            col_penalties.append(np.abs(min1 - min2))

        col_penalties = np.array(col_penalties)
        col_penalties = np.ma.array(col_penalties, mask=np.isnan(col_penalties))
        return col_penalties

    def check_balance(self):
        return np.sum(self.supply) == np.sum(self.demand)

    def get_curr_col_penalties(self):
        return self.col_penalties[-1]

    def get_curr_row_penalties(self):
        return self.row_penalties[-1]

    def get_tariffs_mask(self):
        tariffs_mask = self.tariffs.copy()
        for row in self.skip_rows:
            tariffs_mask[row].fill(np.nan)
        for col in self.skip_cols:
            tariffs_mask[:, col].fill(np.nan)
        return np.ma.masked_where(np.isnan(tariffs_mask), tariffs_mask)

    def find_max_penalty_sum_cell(self):
        col_pens = self.get_curr_col_penalties()
        tariffs_mask = self.get_tariffs_mask()

        max_col_pens = np.where(col_pens == np.nanmax(col_pens))[0]
        sums = []
        for col_idx in max_col_pens:
            col = tariffs_mask.T[col_idx]
            min_tariffs = np.where(col == np.nanmin(col))[0]
            sums.extend(
                (row_idx, col_idx, self.demand[col_idx] + self.supply[row_idx])
                for row_idx in min_tariffs
            )
        return max(sums, key=lambda x: x[2])

    def change_tariffs_n_demands(self, row, col):
        if self.demand[col] > self.supply[row]:
            self.prod_qnts[row][col] = self.supply[row]
            self.demand[col] -= self.supply[row]
            self.supply[row] = 0
            self.skip_rows.append(row)
        else:
            self.prod_qnts[row][col] = self.demand[col]
            self.supply[row] -= self.demand[col]
            self.demand[col] = 0
            self.skip_cols.append(col)

    def stop_iter(self):
        return all(np.isclose(0, i) for i in self.demand) and all(
            np.isclose(0, i) for i in self.supply
        )

    def total_transport_cost(self):
        return np.dot(self.tariffs.flatten(), self.prod_qnts.flatten())

    def solve_ref_plan(self, print_iter=False):
        if not self.check_balance():
            raise ValueError("The problem input parameters are not balanced.")
        if print_iter:
            print(self.header_str())

        while not self.stop_iter():
            self.row_penalties.append(self.find_row_penalties())
            self.col_penalties.append(self.find_column_penalties())
            row, col, _ = self.find_max_penalty_sum_cell()
            self.change_tariffs_n_demands(row, col)

            if print_iter:
                print(self.iter_str())

        return self.total_transport_cost()

    def header_str(self):
        ss = io.StringIO()
        ss.write("=" * 100)
        ss.write("\n")
        ss.write(" " * 40)
        ss.write("Fogel method\n")
        ss.write("=" * 100)
        return ss.getvalue()

    def iter_str(self):
        ss = io.StringIO()
        ss.write(" " * 40)
        ss.write("---=== Iter ===---\n")
        tariffs_mask = self.get_tariffs_mask()

        for i in range(tariffs_mask.shape[0]):
            for j in range(tariffs_mask.shape[1]):
                if tariffs_mask.mask[i][j]:
                    ss.write("!")
                ss.write(f"{self.tariffs[i][j]:.2f}")
                ss.write(f"({self.prod_qnts[i][j]:.2f})")
                ss.write("\t")

            ss.write(f" |\t{self.supply[i]:.2f}\t")
            for row_penalty in self.row_penalties:
                ss.write(f"{row_penalty[i]:.2f}\t")
            ss.write("\n")

        ss.write("-" * 66)
        ss.write("\n")
        for dem in self.demand:
            ss.write(f"{dem:.2f}\t\t")
        ss.write("\n\n")
        for col_penalty in self.col_penalties:
            for pen in col_penalty:
                ss.write(f"{pen:.2f}\t\t")
            ss.write("\n")
        ss.write("\n")
        ss.write(f"L_min = {self.total_transport_cost():.2f}\n")

        return ss.getvalue()


def get_cell_nonzero_neibours(board, source):
    neighbors = []
    s_i, s_j = source
    rows, cols = board.shape

    for i in range(s_i - 1, -1, -1):
        if board.mask[i][s_j]:
            break
        if not np.isclose(board.data[i][s_j], 0):
            neighbors.append((i, s_j))
    for i in range(s_i + 1, rows):
        if board.mask[i][s_j]:
            break
        if not np.isclose(board.data[i][s_j], 0):
            neighbors.append((i, s_j))
            break

    for j in range(s_j - 1, -1, -1):
        if board.mask[s_i][j]:
            break
        if not np.isclose(board.data[s_i][j], 0):
            neighbors.append((s_i, j))
            break
    for j in range(s_j + 1, cols):
        if board.mask[s_i][j]:
            break
        if not np.isclose(board.data[s_i][j], 0):
            neighbors.append((s_i, j))
            break

    return neighbors


def get_graph(board, source):
    s = []
    s.append(source)
    graph = {}
    source_neighbors = get_cell_nonzero_neibours(board, source)

    for neighbor in source_neighbors:
        graph[neighbor] = [source]

    while s:
        curr_cell = s.pop()
        i, j = curr_cell

        if board.mask[i][j]:
            continue

        board.mask[i][j] = True
        neighbors = get_cell_nonzero_neibours(board, curr_cell)

        if curr_cell in graph:
            graph[curr_cell].extend(neighbors)
        else:
            graph[curr_cell] = neighbors

        for neighbor in neighbors:
            s.append(neighbor)

    return graph


def find_looped_path(graph, source, dest, N):
    paths = []

    def dfs(node, path, length):
        if node == dest and length >= N:
            paths.append(path)
            return

        for neighbor in graph[node]:
            if neighbor in path and neighbor == dest and length + 1 >= N:
                dfs(neighbor, path + [neighbor], length + 1)
            elif neighbor not in path:
                dfs(neighbor, path + [neighbor], length + 1)

    dfs(source, [source], 0)
    return min(paths, key=lambda x: len(x)) if paths else None


class PotentialsMethod:
    def __init__(self, tariffs, demand, supply) -> None:
        self.tariffs = np.array(tariffs, dtype=np.float64)
        self.demand = np.array(demand, dtype=np.float64)
        self.supply = np.array(supply, dtype=np.float64)
        self.init_supply = self.supply.copy()
        self.init_demand = self.demand.copy()

        self.transport_volumes = np.zeros((len(supply), len(demand)))
        self.row_potentials = np.zeros(self.supply.size)
        self.col_potentials = np.zeros(self.demand.size)

    def check_balance(self):
        return np.sum(self.supply) == np.sum(self.demand)

    def transform_supply_n_demand(self, row, col):
        md = self.demand[col]
        ms = self.supply[row]
        self.transport_volumes[row][col] = np.min((md, ms))
        self.demand[col], self.supply[row] = exchange_diff(md, ms)

    def build_basic_plan(self):
        min1, min2 = np.sort(np.unique(self.tariffs.flatten()))[:2]

        min1_cells = np.argwhere(self.tariffs == min1)
        min2_cells = np.argwhere(self.tariffs == min2)

        for mcr, mcc in min1_cells:
            self.transform_supply_n_demand(mcr, mcc)
        for m2cr, m2cc in min2_cells:
            self.transform_supply_n_demand(m2cr, m2cc)

        for i in range(self.transport_volumes.shape[0]):
            for j in range(self.transport_volumes.shape[1]):
                if any(i == row and j == col for row, col in min1_cells) or any(
                    i == row and j == col for row, col in min2_cells
                ):
                    continue
                self.transform_supply_n_demand(i, j)

    def check_plan_degeneration(self):
        non_zeros = np.where(self.transport_volumes != 0)[0]
        m, n = self.transport_volumes.shape

        if non_zeros.size == m + n - 1:
            return True
        return False

    def total_transport_cost(self):
        return np.dot(self.tariffs.flatten(), self.transport_volumes.flatten())

    def find_potentials(self):
        u = np.zeros(self.tariffs.shape[0])
        v = np.zeros(self.tariffs.shape[1])
        u.fill(np.nan)
        v.fill(np.nan)
        u[0] = 0

        for i in range(self.tariffs.shape[0]):
            for j in range(self.tariffs.shape[1]):
                c = self.tariffs[i][j]
                if np.isclose(self.transport_volumes[i][j], 0):
                    continue
                if not np.isnan(u[i]) and np.isnan(v[j]):
                    v[j] = c - u[i]
                elif not np.isnan(v[j]) and np.isnan(u[i]):
                    u[i] = c - v[j]

        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0

        return np.array(u), np.array(v)

    def find_worst_infringer(self, u, v):
        infringer_cells = []

        for i in range(self.tariffs.shape[0]):
            for j in range(self.tariffs.shape[1]):
                if not np.isclose(self.transport_volumes[i][j], 0):
                    continue
                sum = u[i] + v[j]
                if sum > self.tariffs[i][j]:
                    infringer_cells.append((i, j, sum - self.tariffs[i][j]))

        if not infringer_cells:
            return None
        return max(infringer_cells, key=lambda x: x[2])

    def improve_plan(self, infringer):
        i, j, *_ = infringer
        graph = get_graph(
            np.ma.masked_array(self.transport_volumes, mask=False),
            (i, j),
        )
        path = find_looped_path(graph, (i, j), (i, j), 3)

        if path is None:
            raise ValueError("No looped path found to optimize the plan.")
        path.pop()  # so we don't update the infringer cell twice

        odd = np.array(
            [self.transport_volumes[i][j] for idx, (i, j) in enumerate(path) if idx % 2]
        )
        alpha = np.min(odd)

        for idx, (i, j) in enumerate(path):
            if idx % 2:
                self.transport_volumes[i][j] -= alpha
            else:
                self.transport_volumes[i][j] += alpha
        return path

    def solve_ref_plan(self, print_iter=False):
        if not self.check_balance():
            raise ValueError("The problem params are not balanced.")
        if print_iter:
            print(self.header_str())

        self.build_basic_plan()

        if print_iter:
            print(self.iter_str())

        if not self.check_plan_degeneration():
            raise ValueError(
                "The basic plan can't be improved as it is not degenerative."
            )
        optimal = False

        while not optimal:
            self.row_potentials, self.col_potentials = self.find_potentials()
            infringer = self.find_worst_infringer(
                self.row_potentials, self.col_potentials
            )

            if infringer:
                path = self.improve_plan(infringer)
            else:
                optimal = True
                path = None

            if print_iter:
                print(self.iter_str(path))

        return self.total_transport_cost()

    def header_str(self):
        ss = io.StringIO()
        ss.write("=" * 70)
        ss.write("\n")
        ss.write(" " * 30)
        ss.write("Potentials method\n")
        ss.write("=" * 70)
        return ss.getvalue()

    def iter_str(self, path=None):
        ss = io.StringIO()
        ss.write(" " * 20)
        ss.write("---=== Iteration ===---\n")

        for i in range(self.transport_volumes.shape[0]):
            for j in range(self.transport_volumes.shape[1]):
                ss.write(str(self.transport_volumes[i][j]))
                ss.write(f"({str(self.tariffs[i][j])})\t")
            ss.write(f"|\t{self.row_potentials[i]}\t")
            ss.write(f"{self.init_supply[i]:.2f}")
            ss.write("\n")
        ss.write("-" * 49)
        ss.write("\n\n")

        for i in range(self.col_potentials.size):
            ss.write(f"{self.col_potentials[i]:.2f}\t\t")
        ss.write("\n")
        for i in range(self.demand.size):
            ss.write(f"{self.init_demand[i]:.2f}\t\t")
        ss.write(f"\n\nL_min = {self.total_transport_cost():.2f}\n")

        if path:
            ss.write("Built chain:\n")
            path_strs = (
                ("(-)" if i % 2 else "(+)") + str(cell) for i, cell in enumerate(path)
            )
            ss.write(" -> ".join(path_strs))
            ss.write("\n")

        return ss.getvalue()


class MinimalElementMethod:
    def __init__(self, tariffs, demand, supply):
        self.tariffs = np.ma.array(tariffs, mask=False, dtype=np.float64)
        self.demand = np.array(demand, dtype=np.float64)
        self.supply = np.array(supply, dtype=np.float64)
        self.transport_volumes = np.zeros((len(supply), len(demand)))

    def find_minimum_element_cells(self):
        min_cells = np.argwhere(self.tariffs == np.ma.min(self.tariffs))
        return min_cells

    def check_balance(self):
        return np.sum(self.supply) == np.sum(self.demand)

    def change_tariffs_n_demands(self, row, col):
        if self.demand[col] > self.supply[row]:
            self.transport_volumes[row][col] = self.supply[row]
            self.demand[col] -= self.supply[row]
            self.supply[row] = 0
            self.tariffs.mask[row] = True
        else:
            self.transport_volumes[row][col] = self.demand[col]
            self.supply[row] -= self.demand[col]
            self.demand[col] = 0
            self.tariffs.mask[:, col] = True

    def total_transport_cost(self):
        return np.dot(self.tariffs.data.flatten(), self.transport_volumes.flatten())

    def stop_iter(self):
        return all(np.isclose(0, i) for i in self.demand) and all(
            np.isclose(0, i) for i in self.supply
        )

    def solve_ref_plan(self, print_iter=False):
        if not self.check_balance():
            raise ValueError("The problem params are not balanced.")
        if print_iter:
            print(self.header_str())
            print(self.iter_str())

        while not self.stop_iter():
            minimum_cells = self.find_minimum_element_cells()

            for i, j in minimum_cells:
                self.change_tariffs_n_demands(i, j)
            if print_iter:
                print(f"Minimum cells: {minimum_cells}")
                print(self.iter_str())

        return self.total_transport_cost()

    def header_str(self):
        ss = io.StringIO()
        ss.write("=" * 70)
        ss.write("\n")
        ss.write(" " * 30)
        ss.write("Minimal element method\n")
        ss.write("=" * 70)
        return ss.getvalue()

    def iter_str(self):
        ss = io.StringIO()
        ss.write(" " * 20)
        ss.write("---=== Iteration ===---\n")

        for i in range(self.tariffs.shape[0]):
            for j in range(self.tariffs.shape[1]):
                ss.write(str(self.tariffs[i][j]))
                ss.write(f"({self.transport_volumes[i][j]:.2f})\t")
            ss.write(f"|\t{self.supply[i]}\n")
        ss.write("-" * 65)
        ss.write("\n\n")

        for i in range(self.demand.size):
            ss.write(f"{self.demand[i]:.2f}\t\t")
        ss.write("\n")
        ss.write(f"L_min = {self.total_transport_cost():.2f}\n")

        return ss.getvalue()


def test_fogel_method():
    tariffs = [
        [5.0, 8, 1, 2],
        [2.0, 5, 4, 9],
        [9.0, 2, 3, 1],
    ]
    demand = [125.0, 90, 130, 100]
    supply = [210.0, 170, 65]

    fm = FogelMethod(tariffs, demand, supply)
    sol = fm.solve_ref_plan(print_iter=True)
    assert sol == 895


def test_potentials_method():
    tariffs = [
        [6.0, 4, 7],
        [8.0, 5, 9],
        [5.0, 6, 8],
        [9.0, 7, 4],
    ]
    demand = [120.0, 100, 110]
    supply = [105.0, 55, 90, 80]

    pm = PotentialsMethod(tariffs, demand, supply)
    transport_cost = pm.solve_ref_plan(print_iter=True)
    assert np.isclose(transport_cost, 1615.0)


def test_minimal_element_method():
    tariffs = [
        [5.0, 8, 1, 2],
        [2.0, 5, 4, 9],
        [9.0, 2, 3, 1],
    ]
    demand = [125.0, 90, 130, 100]
    supply = [210.0, 170, 65]

    fm = MinimalElementMethod(tariffs, demand, supply)
    sol = fm.solve_ref_plan(print_iter=True)
    assert sol == 1100


def test_all_methods_variant16():
    tarifs = [
        [4.0, 5, 6, 6],
        [6.0, 7, 4, 9],
        [7.0, 6, 8, 4],
    ]
    demand = [14.0, 26, 16, 34]
    supply = [30.0, 40, 20]

    fm = FogelMethod(tarifs, demand, supply)
    pm = PotentialsMethod(tarifs, demand, supply)
    mm = MinimalElementMethod(tarifs, demand, supply)

    # fm.solve_ref_plan(print_iter=True)
    # pm.solve_ref_plan(print_iter=True)
    mm.solve_ref_plan(print_iter=True)


def main():
    # test_fogel_method()
    # test_potentials_method()
    # test_minimal_element_method()
    test_all_methods_variant16()


if __name__ == "__main__":
    main()
