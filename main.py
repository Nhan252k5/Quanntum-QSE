from qse.config import  DEFAULT_CONFIG
from qse.pipeline.reduction import run_reduction_pipeline, export_povm
from qse.load_data import load_dataset  


def main():
    
    # rhos = [...]
    #
    # Load dataset của bạn ở đây.
    n, rhos = load_dataset("qse_dataset1_c4_mixed_asymmetric_chain.xlsx")

    config = DEFAULT_CONFIG

    result = (
        run_reduction_pipeline(
            rhos,
            config
        )
    )

    if result.ambiguities:
        print("CẢNH BÁO AMBIGUOUS:", len(result.ambiguities), "quyết định sát ngưỡng")
        for a in result.ambiguities[:10]:
            print("   ", a)

    if result.status == "EXTREMIFY_FAILED":
        print("Extremify không hội tụ.")
        return

    if not result.feasible:
        print(
            "SQSE không khả thi."
        )
        return

    report = result.verification

    print(
        "===== FINAL REPORT ====="
    )

    print(
        "valid =",
        report.valid
    )

    print(
        "extreme =",
        report.extreme
    )

    print(
        "ranks =",
        report.ranks
    )

    print(
        "active =",
        report.active_count
    )

    print(
        "sum r_i^2 =",
        report.sum_rank_squared
    )

    print(
        "D_S =",
        report.D_S
    )

    print(
        "D_K =",
        report.D_K
    )

    print(
        "nullity Phi =",
        report.nullity_phi
    )

    print(
        "completeness residual =",
        report.completeness_residual
    )

    print(
        "exclusion residual =",
        report.exclusion_residual
    )
    if result.status == "OK":
        export_povm(result, "ket_qua_dataset1.xlsx")


if __name__ == "__main__":
    main()