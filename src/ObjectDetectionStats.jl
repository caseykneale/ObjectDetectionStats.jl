module ObjectDetectionStats

    include("LocalizationMeasures.jl")
    export Box, barycenterbox, LLURtoULLR, area, intersection_area,
        intersection_over_union

    include("LocalizationStatistics.jl")
    export HotClassLocalization, ColdClassLocalization,
        ObjectDetectionScore, add!, classwise_precision,
        classwise_recall, precision, recall

end # module
