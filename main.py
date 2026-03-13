import argparse

from cllc_modes.mode_selector import solve_operating_point
from cllc_modes.plotting import plot_mode_result
from cllc_modes.sweep import run_fp_sweep, plot_fp_m_surface, plot_fp_m_by_mode


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--f_min", type=float, default=0.5)
    parser.add_argument("--f_max", type=float, default=1.5)
    parser.add_argument("--f_num", type=int, default=25)
    parser.add_argument("--p_min", type=float, default=0.01)
    parser.add_argument("--p_max", type=float, default=0.8)
    parser.add_argument("--p_num", type=int, default=25)
    parser.add_argument("--sweep_out", type=str, default="fp_m_distribution.png")
    parser.add_argument("--mode_out", type=str, default="fp_m_distribution_by_mode.png")

    args = parser.parse_args()

    if args.sweep:
        records = run_fp_sweep(
            k=args.k,
            f_min=args.f_min,
            f_max=args.f_max,
            f_num=args.f_num,
            p_min=args.p_min,
            p_max=args.p_max,
            p_num=args.p_num,
        )

        valid_count = sum(r["success"] for r in records)
        print(f"Valid points: {valid_count}/{len(records)}")

        plot_fp_m_surface(
            records,
            title=f"(F, P, M) distribution at k={args.k}",
            out_path=args.sweep_out,
        )

        plot_fp_m_by_mode(
            records,
            title=f"(F, P, M) distribution by mode at k={args.k}",
            out_path=args.mode_out,
        )
        return

    result = solve_operating_point(F=args.F, k=args.k, P=args.P)

    print("mode:", result.mode)
    print("success:", result.success)
    print("max_residual:", result.max_residual)
    print("params:")
    for kk, vv in result.params.items():
        print(f"  {kk} = {vv}")

    plot_mode_result(result, out_path=args.out)


if __name__ == "__main__":
    main()