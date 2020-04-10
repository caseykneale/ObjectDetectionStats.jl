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

# FOR EACH CLASS:
#
# sorted by decreasing confidence and assigned to ground-truth objects.
#     - We have "a match" when they share the same label and an IoU >= 0.5
#     - This "match" is considered a TP if that GT has not been already used
#
# Then we compute a version of the measured PR curve with P monotonically decreasing
# by setting the precision for recall r to the maximum precision obtained for any recall r' > r.
#
#DataFrames are a better Data Structure here.. time for a rewrite!
