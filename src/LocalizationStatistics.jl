struct HotClassLocalization
    length::Int
    classes::Int
    scores::Matrix{Float64}
    locations::Vector{ Box }
end

"""
    HotClassLocalization(scores::Matrix{Float64, 2}, locations::Vector{ Box })

"""
function HotClassLocalization(scores::Matrix{Float64}, locations::Vector{ Box })
    len, classes = size(scores)
    @assert len == length(locations)
    return HotClassLocalization( len, classes, scores, locations )
end

struct ColdClassLocalization
    class_location::Vector{ Pair{ Int, Box } }
end

"""
    ColdClassLocalization( cold_encodings::Vector{Int}, locations::Vector{Box} )

"""
function ColdClassLocalization( cold_encodings::Vector{Int}, locations::Vector{Box} )
    return ColdClassLocalization( cold_encodings .=> locations )
end

mutable struct ObjectDetectionScore
    TP::Vector{Int}
    FP::Vector{Int}
    FN::Vector{Int}
    IoU_threshold::Float64
    score_threshold::Float64
end

function ObjectDetectionScore(class_no)
    return ObjectDetectionScore(    zeros(Int, class_no),
                                    zeros(Int, class_no),
                                    zeros(Int, class_no),
                                    0.5, 0.5 )
end

function ObjectDetectionScore( class_no, IoU_eps, score_eps )
    return ObjectDetectionScore(    zeros(Int, classes),
                                    zeros(Int, classes),
                                    zeros(Int, classes),
                                    IoU_eps, score_eps )
end

"""
    add!( ods::ObjectDetectionScore, tp, fp, fn )

Update an ObjectDetectionScore instance with scores from an image/batch.

"""
function add!( ods::ObjectDetectionScore, tp, fp, fn )
    ods.TP .+= tp; ods.FP .+= fp; ods.FN .+= fn
    return nothing
end

"""
    (ods::ObjectDetectionScore)( predictions::HotClassLocalization, GT::ColdClassLocalization )

Use an ObjectDetectionScore instance to store prediction statistics.

"""
function (ods::ObjectDetectionScore)(   predictions::HotClassLocalization,
                                        GT::ColdClassLocalization )
    #Find highest prediction of every box
    cold_preds      = [ argmax( predictions.scores[p,:] ) for p in 1:predictions.length ]
    positive_preds  = [ predictions.scores[row,p] > ods.score_threshold for ( row, p ) in enumerate( cold_preds ) ]
    pos_inds        = findall(positive_preds)
    total_positives = sum( positive_preds )
    Pos             = [ false for _ in 1:total_positives ]
    #Assume classwise distinction is not valuable here...
    #Find the best match for a given predictor to a given GT
    for ( gt_cold, gt_box ) in GT.class_location
        best_IoU, best_ind = 0.0, missing
        for ( i, p ) in enumerate( pos_inds )
            #Is this a suitable prediction? Avoid double dipping
            if ( !Pos[ i ] ) && ( cold_preds[p] == gt_cold )
                #find best overlap...
                iou = intersection_over_union( gt_box, predictions.locations[ p ] )
                if iou > best_IoU
                    best_IoU, best_ind = iou, p
                end
            end
        end
        if !ismissing(best_ind)
            Pos[best_ind] = best_IoU > 0.0
        end
    end
    class_TPs, class_FPs, class_FNs = [ zeros(Int, predictions.classes) for _ in 1:3 ]
    for (i, class) in enumerate( 1:predictions.classes )
        class_inds = findall( first.( GT.class_location ) .== class )
        predicted_positives = findall( cold_preds[ pos_inds ] .== class)
        incorrect_pos = .!Pos[ predicted_positives ]

        class_FPs[i] = sum(incorrect_pos)
        if length( class_inds ) > 0
            class_TPs[i] = sum( Pos[class_inds] )
            class_FNs[i] = length( class_inds ) - class_TPs[i]
        end
    end
    add!( ods, class_TPs, class_FPs, class_FNs )
end

"""
    classwise_precision(ods::ObjectDetectionScore)

Returns the classwise precision of an ObjectDetectionScore instance.

"""
classwise_precision(ods::ObjectDetectionScore) = ods.TP ./ ( ods.TP .+ ods.FP )

"""
    classwise_recall(ods::ObjectDetectionScore)

Returns the classwise recall of an ObjectDetectionScore instance.

"""
classwise_recall( ods::ObjectDetectionScore ) = ods.TP ./ ( ods.TP .+ ods.FN )


"""
    precision(ods::ObjectDetectionScore)

Returns the precision of an ObjectDetectionScore instance of all classes.

"""
function precision(ods::ObjectDetectionScore)
    tps, fps = sum( ods.TP ), sum( ods.FP )
    return tps ./ ( tps .+ fps )
end

"""
    recall(ods::ObjectDetectionScore)

Returns the precision of an ObjectDetectionScore instance of all classes.

"""
function recall(ods::ObjectDetectionScore)
    tps, fns = sum( ods.TP ), sum( ods.FN )
    return tps ./ ( tps .+ fns )
end
