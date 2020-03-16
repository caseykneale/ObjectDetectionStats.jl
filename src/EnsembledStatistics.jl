mutable struct
    scores::Vector{ Float64 }
    precision_at_recall::Matrix{ ObjectDetectionScore }
end

function AveragePrecision(  predictions::HotClassLocalization,
                            GT::ColdClassLocalization;
                            k = 0:0.1:1.0)
    # class_no = predictions.classes
    #
    # ObjectDetectionScore( class_no, IoU_eps, score_eps )
    #
    #
    # #prepare inputs for evaluation...
    # hcl = HotClassLocalization( pred_scores, pred_locations )
    # ccl = ColdClassLocalization( GT_cold_encodings, pred_locations )
    # ods_machine( hcl, ccl )
    # @test all(ods_machine.TP .== [0,3,0])
    # @test all(ods_machine.FP .== [0,0,0])
    # @test all(ods_machine.FN .== [0,0,0])
end
