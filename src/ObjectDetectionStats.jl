module ObjectDetectionStats

    include("LocalizationMeasures.jl")
    export Box, ==, barycenterbox, LLURtoULLR, translate, translate!,
        clamp, clamp!, area, intersection_area, intersection_over_union

    include("LocalizationStatistics.jl")
    export HotClassLocalization, ColdClassLocalization, ObjectDetectionScore,
        add!, micro_precision, micro_recall, micro_F1, macro_precision,
        macro_recall, macro_F1, classwise_PR_estimate

end # module
