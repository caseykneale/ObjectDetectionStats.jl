using Documenter, ObjectDetectionStats

makedocs(;
    modules=[ObjectDetectionStats],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/caseykneale/ObjectDetectionStats.jl/blob/{commit}{path}#L{line}",
    sitename="ObjectDetectionStats.jl",
    authors="Casey Kneale",
    assets=String[],
)

deploydocs(;
    repo="github.com/caseykneale/ObjectDetectionStats.jl",
)
