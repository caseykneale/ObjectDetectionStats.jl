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

Use an ObjectDetectionScore instance to calculate and store prediction statistics for a single image.

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
        best_score, best_ind = 0.0, missing
        for ( i, p ) in enumerate( pos_inds )
            #Is this a suitable prediction? Avoid double dipping
            if ( !Pos[ i ] ) && ( cold_preds[p] == gt_cold )
                #is overlapped...
                iou = intersection_over_union( gt_box, predictions.locations[ p ] )
                if (iou >= ods.IoU_threshold) && (positive_preds[ i ] > best_score)
                    best_score, best_ind = positive_preds[ i ], p
                end
            end
        end
        if !ismissing(best_ind)
             Pos[best_ind] = true
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
    classwise_F1(ods::ObjectDetectionScore)

Returns the classwise F1 measure of an ObjectDetectionScore instance.

"""
function classwise_F1( ods::ObjectDetectionScore )
    p = classwise_precision(ods)
    r = classwise_recall(ods)
    return 2.0 * ( (p .* r) ./ (p .+ r) )
end

"""
    precision(ods::ObjectDetectionScore)

Returns the precision of an ObjectDetectionScore instance of all classes.

"""
function macro_precision(ods::ObjectDetectionScore)
    tps, fps = sum( ods.TP ), sum( ods.FP )
    return tps ./ ( tps .+ fps )
end

"""
    recall(ods::ObjectDetectionScore)

Returns the precision of an ObjectDetectionScore instance of all classes.

"""
function macro_recall(ods::ObjectDetectionScore)
    tps, fns = sum( ods.TP ), sum( ods.FN )
    return tps ./ ( tps .+ fns )
end

"""
    classwise_F1(ods::ObjectDetectionScore)

Returns the F1 measure of an ObjectDetectionScore instance of all classes.

"""
function macro_F1( ods::ObjectDetectionScore )
    p = macro_precision(ods)
    r = macro_recall(ods)
    return 2.0 * ( (p .* r) ./ (p .+ r) )
end

"""
    classwise_PR_estimate(  predictions::HotClassLocalization,
                        GT::ColdClassLocalization;
                        IoU = 0.5, K = 100 )

Estimates a PR curve for all classes(columnwise) at K confidence levels (bound between 0 and 1),
at a single intersection over union threshhold (IoU).

Note: this is not a true PR curve where only the values of the predicted scores are used.
This provides a more coarse, but faster to compute version.
"""
function classwise_PR_estimate(  predictions::HotClassLocalization,
                        GT::ColdClassLocalization;
                        IoU = 0.5, K = 100 )
    class_no = GT.class_no
    precisions, recalls = zeros(K, class_no), zeros(K, class_no)
    for (i, interval) in enumerate( 0 : ( 1.0 / (K - 1) ) : 1.0 )
        ods = ObjectDetectionScore( class_no, IoU, interval )
        ods( predictions, GT )
        precisions[i,:] = classwise_precision( ods )
        recalls[i,:] = classwise_recall( ods )
    end
    return precisions, recalls
end

"""
    classwise_PR(  predictions::HotClassLocalization,
                        GT::ColdClassLocalization;
                        IoU = 0.5 )

Generates PR curves for all classes(columnwise) at K confidence levels (bound between 0 and 1),
at a single intersection over union threshhold (IoU).
"""
function classwise_PR(  predictions::HotClassLocalization,
                        GT::ColdClassLocalization;
                        IoU = 0.5, eps = 1e-4 )
    class_no = GT.class_no
    digits = convert(Int, round( -log10( eps ) ) )
    scores = unique.( round.(predictions.scores[:], digits = digits ) )
    K = length(scores)
    precisions, recalls = zeros(K, class_no), zeros(K, class_no)
    for (i, interval) in enumerate(scores)
        ods = ObjectDetectionScore( class_no, IoU, interval )
        ods( predictions, GT )
        precisions[i,:] = classwise_precision( ods )
        recalls[i,:] = classwise_recall( ods )
    end
    return precisions, recalls
end



#Image -> GT( x1,y1,x2,y2 ) + prediction( scores matrix ) + prediction( locations )
#Data -> 1000x Images
#To calculate PR
#Positive predictions are first considered
#       Loop over unique score values
#       IoU's must be stored
#       The highest vote will always win!
#
