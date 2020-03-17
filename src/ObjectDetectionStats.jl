module ObjectDetectionStats

    include("LocalizationMeasures.jl")
    export Box, barycenterbox, LLURtoULLR, area, intersection_area,
        intersection_over_union

    include("LocalizationStatistics.jl")
    export HotClassLocalization, ColdClassLocalization,
        ObjectDetectionScore, add!, classwise_precision,
        classwise_recall, classwise_F1, macro_precision,
        macro_recall, macro_F1, classwise_PR_estimate

    #include("EnsembledStatistics.jl")
    #export

end # module
