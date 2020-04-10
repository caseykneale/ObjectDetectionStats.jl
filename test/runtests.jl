#!/usr/bin/env julia
using ObjectDetectionStats
using Test

@testset "Basic Data Ops" begin
    z = DataFrame( (:a => [1,2], :B => [120,13], :c => [111,111]))
    @test highest_vote!(z, [:a,:B,:c])[:Highest_Vote] .== [ :B, :c ]

    barycenter = DataFrame( Dict( [   :notneeded    => ["a", "b"],
                                      :center_x     => [15, 15],
                                      :center_y     => [10, 10],
                                      :width        => [5, 10],
                                      :height       => [5, 10]
                                  ]) )

    @test has_required_columns(barycenter, [ :center_x, :center_y, :width, :height ]  )
    @test !has_required_columns(barycenter, [ :center_x, :center_y, :width, :height, :NOPE ]  )
    #Test DF for comparing all others with
    KeyDF = DataFrame( Dict( [    :notneeded        => ["a", "b"],
                                  :upper_left_x     => [12.5, 10.0],
                                  :lower_right_x    => [17.5, 20.0],
                                  :upper_left_y     => [7.50, 5.00],
                                  :lower_right_y    => [12.5, 15.0]
                             ] ) )
    #Convert barycenter boxes to ULLR
    barycenter_boxes!( barycenter )
    @test all( Matrix(KeyDF) .== Matrix(barycenter[!, names(KeyDF)]) )

    area!(barycenter)
    @test all( barycenter[ :Area ] .== [ 25, 100 ] )

end

@testset "Box Translate and Clamp Tests" begin
    a            = Box(5,5,15,15)
    trans_result = translate( a, -5, -5 )
    @test trans_result                          == Box(0,0,10,10)
    @test ObjectDetectionStats.clamp(trans_result, Box(5,5,15,15))   == Box(5,5,10,10)
    @test ObjectDetectionStats.clamp(trans_result, Box(0,0, 5, 5))   == Box(0,0, 5, 5)
end

@testset "Box Area Tests" begin
    @test area( Box(5,5,15,15) ) == 100
    @test area( Box(5,5,11,11) ) == 36
end

@testset "Box Intersection Tests" begin
    a = Box(5,5,15,15)
    b = Box(5,5,15,15)
    @test intersection_area( a, b ) == 100
    @test intersection_over_union( a, b ) == 1.0

    a = Box(10,10,15,15)
    b = Box(5,5,15,15)
    @test intersection_area( a, b ) == 25
    @test intersection_over_union( a, b ) == 0.25

    b = Box(10,10,15,15)
    a = Box(5,5,15,15)
    @test intersection_area( a, b ) == 25
    @test intersection_over_union( a, b ) == 0.25

    a = Box(1,1,15,15)
    b = Box(25,25,35,35)
    @test intersection_area( a, b ) == 0
    @test intersection_over_union( a, b ) == 0.0
end

@testset "3 TP test" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.9 0.5 ; #3
                        0.2 0.9 0.5;  #2
                        0.5 0.9 0.2]  #1
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 15,    1,  25,     20 ),
                        Box( 1,    15,  10,     25 ) ]
    GT_cold_encodings = [ 2, 2, 2 ]# all correct
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, pred_locations )
    ods_machine( hcl, ccl )
    @test all(ods_machine.TP .== [0,3,0])
    @test all(ods_machine.FP .== [0,0,0])
    @test all(ods_machine.FN .== [0,0,0])
end

@testset "3 wrong class test" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.9 0.5 ; #3
                        0.2 0.9 0.5; #2
                        0.5 0.9 0.2]  #1
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 15,    1,  25,     20 ),
                        Box( 1,    15,  10,     25 ) ]
    GT_cold_encodings = [ 1, 1, 1 ]# all incorrect class but perfect overlap
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, pred_locations )

    ods_machine( hcl, ccl )
    @test all(ods_machine.TP .== [0,0,0])
    @test all(ods_machine.FP .== [0,3,0])
    @test all(ods_machine.FN .== [3,0,0])
end

@testset "3 wrong class locations with correct predictions" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.9 0.5 ; #2
                        0.2 0.9 0.5;  #2
                        0.5 0.9 0.2 ] #2
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 15,    1,  25,     20 ),
                        Box( 1,    15,  10,     25 ) ]
    GT_cold_encodings = [ 2, 2, 2 ]# all correct class
    GT_locations  = [   Box( 100,    100,  100,     100 ),
                        Box( 150,    100,  250,     200 ),
                        Box( 100,    150,  100,     250 ) ]
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, GT_locations )

    ods_machine( hcl, ccl )

    @test all(ods_machine.TP .== [0,0,0])
    @test all(ods_machine.FP .== [0,3,0])
    @test all(ods_machine.FN .== [0,3,0])
end

@testset "3 wrong class locations with wrong predictions" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.5 0.9; #3
                        0.2 0.9 0.5; #2
                        0.9 0.5 0.2] #1
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 15,    1,  25,     20 ),
                        Box( 1,    15,  10,     25 ) ]
    GT_cold_encodings     = [   3, #correct
                                1, #incorrect
                                1 ]#correct
    GT_locations  = [   Box( 1,     1,  10,     10 ), #TP class 3
                        Box( 15,    1,  25,     20 ), #FP class 1
                        Box( 30,    30, 42,     42 )] #FN class 1
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, GT_locations )
    ods_machine( hcl, ccl )
    @test all(ods_machine.TP .== [0,0,1])
    @test all(ods_machine.FP .== [1,1,0])
    @test all(ods_machine.FN .== [2,0,0])
end

@testset "2 correct overlapping a prediction and 1 nonprediction" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.9 0.5 ; #2
                        0.2 0.9 0.5;  #2
                        0.0 0.0 0.0;] #0 not predicted
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 1,     1,  10,     10 ),
                        Box( 15,   15,  20,     20 )  ]
    GT_cold_encodings = [ 2 ]# all correct class
    GT_locations  = [   Box( 1,     1,  10,     10 ) ]
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, GT_locations )

    ods_machine( hcl, ccl )
    @test all(ods_machine.TP .== [0,1,0])
    @test all(ods_machine.FP .== [0,1,0])
    @test all(ods_machine.FN .== [0,0,0])
end
