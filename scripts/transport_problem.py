import numpy as np
import io


class FogelMethod:
    def __init__(
        self,
        tariffs,
        demand,
        stock,
    ) -> None:
        self.tariffs = np.array(tariffs)
        self.prod_qnts = np.zeros((len(stock), len(demand)))
        self.stock = np.array(stock)
        self.demand = np.array(demand)

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
        return np.sum(self.stock) == np.sum(self.demand)

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
                (row_idx, col_idx, self.demand[col_idx] + self.stock[row_idx])
                for row_idx in min_tariffs
            )
        return max(sums, key=lambda x: x[2])

    def change_tariffs_and_demands(self, row, col):
        if self.demand[col] > self.stock[row]:
            self.prod_qnts[row][col] = self.stock[row]
            self.demand[col] -= self.stock[row]
            self.stock[row] = 0
            self.skip_rows.append(row)
        else:
            self.prod_qnts[row][col] = self.demand[col]
            self.stock[row] -= self.demand[col]
            self.demand[col] = 0
            self.skip_cols.append(col)

    def stop_iter(self):
        return all(np.isclose(0, i) for i in self.demand) and all(
            np.isclose(0, i) for i in self.stock
        )

    def solve_ref_plan(self, print_iter=False):
        while not self.stop_iter():
            self.row_penalties.append(self.find_row_penalties())
            self.col_penalties.append(self.find_column_penalties())
            row, col, _ = self.find_max_penalty_sum_cell()
            self.change_tariffs_and_demands(row, col)

            if print_iter:
                print(self.iter_str())

        return np.dot(self.tariffs.flatten(), self.prod_qnts.flatten())

    def iter_str(self):
        ss = io.StringIO()
        ss.write("---=== Iter ===---\n")
        tariffs_mask = self.get_tariffs_mask()

        for i in range(tariffs_mask.shape[0]):
            for j in range(tariffs_mask.shape[1]):
                ss.write(f"{tariffs_mask[i][j]:.2f}")
                ss.write("\t")

            ss.write(f" |\t{self.stock[i]:.2f}\t")
            for row_penalty in self.row_penalties:
                ss.write(f"{row_penalty[i]:.2f}\t")
            ss.write("\n")

        ss.write("-" * 34)
        ss.write("\n")
        for dem in self.demand:
            ss.write(f"{dem:.2f}\t")
        ss.write("\n\n")
        for col_penalty in self.col_penalties:
            for pen in col_penalty:
                ss.write(f"{pen:.2f}\t")
            ss.write("\n")
        ss.write("\n")

        for i in range(self.prod_qnts.shape[0]):
            for j in range(self.prod_qnts.shape[1]):
                ss.write(f"{self.prod_qnts[i][j]:.2f}\t")
            ss.write("\n")
        ss.write("\n")

        return ss.getvalue()


def test_fogel_method():
    tariffs = [
        [5.0, 8, 1, 2],
        [2.0, 5, 4, 9],
        [9.0, 2, 3, 1],
    ]
    demand = [125.0, 90, 130, 100]
    stock = [210.0, 170, 65]

    fm = FogelMethod(tariffs, demand, stock)
    sol = fm.solve_ref_plan(print_iter=True)
    print(f"L(X_ф) = {sol}")
    assert sol == 895

def main():
    test_fogel_method()


if __name__ == "__main__":
    main()
