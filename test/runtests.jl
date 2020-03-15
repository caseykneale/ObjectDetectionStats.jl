#!/usr/bin/env julia
using ObjectDetectionStats
using Test

@testset "Areas" begin
    @test area( Box(5,5,15,15) ) == 100
    @test area( Box(5,5,11,11) ) == 36
end

@testset "Intersections" begin
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
