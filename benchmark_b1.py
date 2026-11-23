
from lander.params import P
from lander.lcvx import (
    LanderSOCP,
    solve_min_fuel,
    solve_min_landing_error,
)
from lander.pointmass import run_b1


def run_benchmark(certify=True):

    # Build the SOCP once
    prob = LanderSOCP(P, N=40)

    def guidance(x0, m0):
        # First attempt: minimum-fuel landing
        sol, _ = solve_min_fuel(
            prob, x0, m0=m0, certify=certify
        )

        if sol is not None:
            return sol

        # Fallback: minimum landing error
        sol, d_star, _ = solve_min_landing_error(
            P, x0, m0=m0, N=40, certify=certify
        )
        return sol

    def d_star_fn(x0, m0):
        _, d_star, _ = solve_min_landing_error(
            P, x0, m0=m0, N=40, certify=certify
        )
        return d_star

    rows = run_b1(guidance, d_star_fn)

    print(f"\ncertify={certify}")
    print(f"Successful landings: "
          f"{sum(r['landed'] for r in rows)}/20")

    for row in rows:
        print(row)

    return rows


if __name__ == "__main__":
    rows_uncertified = run_benchmark(certify=False)
    rows_certified = run_benchmark(certify=True)
