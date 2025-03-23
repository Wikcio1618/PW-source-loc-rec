module MainModule
include("Evaluation.jl")
export evaluate_original_to_file,
    evaluate_modify_to_file,
    evaluate_reconstruct_to_file,
    calc_auc,
    LPTVA_loc,
    GMLA_loc

end