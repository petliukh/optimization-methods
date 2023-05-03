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
        self.stock = np.array(stock)
        self.demand = np.array(demand)

        self.skip_rows = []
        self.skip_cols = []

    def find_row_penalties(self):
        row_penalties = []

        for i, row in enumerate(self.tariffs):
            if i in self.skip_rows:
                row_penalties.append(np.nan)
                continue

            a, b = np.partition(row, 1)[0:2]
            row_penalties.append(np.abs(a - b))

        row_penalties = np.array(row_penalties)
        row_penalties = np.ma.array(row_penalties, mask=np.isnan(row_penalties))
        return row_penalties

    def find_column_penalties(self):
        col_penalties = []
        for j, col in enumerate(self.tariffs.T):
            if j in self.skip_cols:
                col_penalties.append(np.nan)
                continue

            a, b = np.partition(col, 1)[0:2]
            col_penalties.append(np.abs(a - b))

        col_penalties = np.array(col_penalties)
        col_penalties = np.ma.array(col_penalties, mask=np.isnan(col_penalties))
        return col_penalties

    def check_balance(self):
        return np.sum(self.stock) == np.sum(self.demand)

    def solve_ref_plan(self, print_iter=False):
        ...

    def __str__(self):
        ss = io.StringIO()
        ss.write("---=== Iter ===---")

        for i in range(self.tariffs.shape[0]):
            for j in range(self.tariffs.shape[1]):
                ss.write(str(self.tariffs[i][j]))
                ss.write("\t")
                ss.write(str(self.stock[i]))
            ss.write("\n")
        for i in range(self.demand.size):
            ss.write(str(self.demand[i]))
            ss.write("\t")
        ss.write("\n")

        return ss.getvalue()


def main():
    tariffs = [
        [5.0, 8, 1, 2],
        [2.0, 5, 4, 9],
        [9.0, 2, 3, 1],
    ]
    demand = [125.0, 90, 130, 100]
    stock = [210.0, 170, 65]

    fm = FogelMethod(tariffs, demand, stock)
    fm.solve_ref_plan(print_iter=True)


if __name__ == "__main__":
    main()
