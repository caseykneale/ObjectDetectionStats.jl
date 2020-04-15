using Pkg
Pkg.activate(".")
using DataFrames

global ULLR_required_columns = [ :upper_left_x, :lower_right_x, :upper_left_y, :lower_right_y ]

"""
    has_required_columns( df::DataFrame, required_columns::Vector )::Bool

Returns `true` if a `DataFrame` instance has all of the `required_columns` present.

"""
function has_required_columns( df::DataFrame, required_columns::Vector )::Bool
    df_names = names( df )
    result = true
    for required_column in required_columns
        if !( required_column in df_names )
            result = false
            break
        end
    end
    return result
end

"""
    highest_vote!( df::DataFrame, class_columns::Vector )::DataFrame

Finds highest vote amongst all specified class columns and creates a new column with that variable name.

"""
function highest_vote!( df::DataFrame, class_columns::Vector )::DataFrame
    df_classes = df[ !, class_columns ]
    df[ !, :Highest_Vote ] = map( row -> argmax( row ), eachrow( df_classes ) )
    df_classes = df[ !, vcat( class_columns, :Highest_Vote ) ]
    df[ !, :Highest_Prediction ] = map( row -> row[row.Highest_Vote], eachrow( df_classes ) )
    return df
end

highest_vote!( dfs::Vector{DataFrame}, class_columns::Vector )::Vector{DataFrame} = [ highest_vote!(df, class_columns) for df in dfs ]

"""
    barycenter_boxes!( df::DataFrame )::DataFrame

Will convert a DataFrame containing the following variables: center_x, center_y, width, height to be formatted in upper left lower right coordinates.

"""
function barycenter_boxes!( df::DataFrame )::DataFrame
    req_cols = [ :center_x, :center_y, :width, :height ]
    @assert( has_required_columns( df, req_cols ), "Input dataframe does not have the required columns: $req_cols. Please check spelling and capitalization of column names." )
    df[ !, [ :width, :height] ] ./= 2
    df[ !, :upper_left_x ] = df.center_x - df.width
    df[ !, :lower_right_x] = df.center_x + df.width
    df[ !, :upper_left_y ] = df.center_y - df.height
    df[ !, :lower_right_y] = df.center_y + df.height
    select!( df, Not( req_cols ) )
    return df
end

"""
    LLUR_boxes!( df::DataFrame )::DataFrame
"""
function LLUR_boxes!( df::DataFrame )::DataFrame
    req_cols = [ :lower_left_x, :upper_right_x, :lower_left_y, :upper_right_y ]
    new_cols = ULLR_required_columns
    @assert( has_required_columns( df, req_cols ), "Input dataframe does not have the required columns: $req_cols. Please check spelling and capitalization of column names." )
    rename!( df, Dict( req_cols .=> new_cols ) )
    return df
end

"""
    area( df::DataFrame )::DataFrame

Calculates the areas of all Boxes in a DataFrame as a new column.

"""
function area!( df::DataFrame )::DataFrame
    @assert( has_required_columns( df, ULLR_required_columns ), "Input dataframe does not have the required columns: $ULLR_required_columns. Please check spelling and capitalization of column names and ensure the format is ULLR." )
    df[ !, :Area ] = ( df.lower_right_x - df.upper_left_x ) .* ( df.lower_right_y - df.upper_left_y )
    return df
end

area!( dfs::Vector{DataFrame} )::Vector{DataFrame} = [ area!(df) for df in dfs ]

"""
    intersection_area( a::DataFrame, b::DataFrame )::DataFrame

Calculates the area of intersection between 2 DataFrames containing ULLR boxes.
This returns a new DataFrame

"""
function intersection_over_union( a::DataFrame, b::DataFrame )::Tuple{DataFrame,DataFrame}
    @assert( has_required_columns( a, ULLR_required_columns ), "Input dataframe does not have the required columns: $ULLR_required_columns. Please check spelling and capitalization of column names and ensure the format is ULLR." )
    @assert( has_required_columns( b, ULLR_required_columns ), "Input dataframe does not have the required columns: $ULLR_required_columns. Please check spelling and capitalization of column names and ensure the format is ULLR." )
    @assert( has_required_columns( a, [:Area] ), "Input dataframe does not have the required columns: Area. Please run `area()` on the individual dataframes." )
    @assert( has_required_columns( b, [:Area] ), "Input dataframe does not have the required columns: Area. Please run `area()` on the individual dataframes." )
    len_a, len_b = first.(size.( [ a, b ] ) )
    total_len = len_a * len_b

    output = DataFrame( (   :ID_A => Vector{Int}( undef, total_len ),
                            :ID_B => Vector{Int}( undef, total_len ),
                            :class_labels => Vector{Symbol}( undef, total_len ),
                            :Intersection_Area => Vector{Float64}( undef, total_len ),
                            :Intersection_Over_Union => Vector{Float64}( undef, total_len ),
                            :Vote_Matches => Vector{Bool}( undef, total_len ),
                            :Highest_Prediction => Vector{Float64}( undef, total_len),
                            :Highest_Vote => Vector{Symbol}( undef, total_len)
                        ) )
    current_row = 1
    for ( row_a_ID, row_a ) in enumerate( eachrow( a ) ), ( row_b_ID, row_b ) in enumerate( eachrow( b ) )
        #current_row = row_a_ID + ( len_a * ( row_b_ID - 1 ) )
        output[ current_row, :ID_A ] = row_a_ID
        output[ current_row, :ID_B ] = row_b_ID
        output[ current_row, :class_labels ] = row_b.class_labels
        output[ current_row, :Vote_Matches ] = row_a.Highest_Vote == row_b.class_labels
        output[ current_row, :Highest_Prediction ] = row_a.Highest_Prediction #row_b[!,row_a.Highest_Vote]
        output[ current_row, :Highest_Vote ] = row_a.Highest_Vote
        output[ current_row, :Intersection_Area ] = max( 0, min( row_a.lower_right_x, row_b.lower_right_x ) - max( row_a.upper_left_x, row_b.upper_left_x ) ) .*
                                                    max( 0, min( row_a.lower_right_y, row_b.lower_right_y ) - max( row_a.upper_left_y, row_b.upper_left_y ) )
        output[ current_row, :Intersection_Over_Union ] = output[ current_row, :Intersection_Area ] / ( row_a.Area + row_b.Area - output[ current_row, :Intersection_Area ])
        current_row += 1
    end
    #remove intermediate calculation column Intersection_Area
    select!(output, Not(:Intersection_Area))
    #Split data by each unique prediction
    #   -> sort by intersection over union (for each prediction all others are constant)
    #   -> grab the highest IoU and toss the rest.
    result = DataFrame( map( x -> sort(x, :Intersection_Over_Union, rev = true)[1,:], groupby( output, :ID_A ) ) )
    #Find all GT's that have no matching predictions
    possible_FNs = 1:first( size( b ) )
    not_predicted = [ !( p in result.ID_B ) for p in possible_FNs ]
    FNs = sum(not_predicted)
    definite_FNs = possible_FNs[ not_predicted ]
    FN_df = DataFrame( (    :ID_A => zeros( Int, FNs ),
                            :ID_B => definite_FNs,
                            :class_labels => b.class_labels[definite_FNs],
                            :Intersection_Over_Union => zeros( Float64, FNs ),
                            :Vote_Matches => repeat( [false], FNs ),
                            :Highest_Prediction => zeros(Float64, FNs ),
                            :Highest_Vote => repeat([:none], FNs)
                        ) )
    return result, FN_df
end

"""
    potential_scores( ods::ObjectDetectionStats; IoU_threshold = 0.5 )

"""
function potential_scores( cur_IoU_df::DataFrame; IoU_threshold = 0.5 )
    cur_IoU_df[ !, :iou_threshold ] = cur_IoU_df.Intersection_Over_Union .> IoU_threshold
    cur_IoU_df[ !, :TP_or_FN ] = cur_IoU_df.Vote_Matches .& cur_IoU_df.iou_threshold
    #Ensure we do not double dip from a GT prediction. Only one prediction can map to a GT here...
    #Group by ground truth class label
    #   -> sort by the highest predicted class score
    #       -> find first predicted class matches above an IoU threshold
    for gdf in groupby( cur_IoU_df, :ID_B )
        sort!( cur_IoU_df, :Highest_Prediction, rev = true)
        kept_first = false
        for row in eachrow( gdf )
            if row.TP_or_FN
                if kept_first
                    row.TP_or_FN = false
                end
                kept_first = true
            end
        end
    end
    #Bifurcate DF on the possibility of true positives
    TP_or_FN, FP_or_TN = DataFrame.( collect( groupby( cur_IoU_df, :TP_or_FN ) ) )
    return TP_or_FN, FP_or_TN
end

mutable struct ObjectDetectionStats
    TP_or_FN::Union{ Missing, DataFrame }
    FP_or_TN::Union{ Missing, DataFrame }
    FN::Union{ Missing, DataFrame }
    classnames::Vector{ Symbol }
    IoU_threshold::Float64
end

"""
    ObjectDetectionStats( classnames::Vector{Symbol}; IoU_threshold = 0.5 )

Instantiates an `ObjectDetectionStats` object.

"""
function ObjectDetectionStats( classnames::Vector{Symbol}; IoU_threshold = 0.5 )
    return ObjectDetectionStats( missing, missing, missing, classnames, IoU_threshold )
end

"""
    (ods::ObjectDetectionStats)( predictions::DataFrame, groundtruth::DataFrame )

Add an example to an `ObjectDetectionStats` instance and evaluate its scores.

"""
function (ods::ObjectDetectionStats)( predictions::DataFrame, groundtruth::DataFrame)
    area!( predictions )
    highest_vote!( predictions, ods.classnames )
    area!( groundtruth )
    IoU_map, FN_df = intersection_over_union( predictions, groundtruth )
    TP_or_FN, FP_or_TN = potential_scores( IoU_map; IoU_threshold = ods.IoU_threshold )
    ods.TP_or_FN = ismissing(ods.TP_or_FN) ? TP_or_FN : vcat( ods.TP_or_FN, TP_or_FN )
    ods.FP_or_TN = ismissing(ods.FP_or_TN) ? FP_or_TN : vcat( ods.FP_or_TN, FP_or_TN )
    ods.FN = ismissing( ods.FN ) ? FN_df : vcat( ods.FN, FN_df )
    #find definitive false negatives: IE GT instances without any mappable predictions
    return nothing
end

using DataFrames

KeyDF = DataFrame( Dict( [    :a               => [0.8, 0.8, 0.1, 0.1],
                              :b               => [0.1, 0.1, 0.7, 0.1],
                              :c               => [0.1, 0.1, 0.2, 0.8],
                              :upper_left_x     => [1., 1.1, 4., 7.],
                              :lower_right_x    => [2., 2.0, 5., 8.],
                              :upper_left_y     => [1., 1.1, 1., 7.],
                              :lower_right_y    => [2., 2.0, 5., 8.]
                         ] ) )

#4th b has no matches
GT = DataFrame( Dict( [     :class_labels     => [:a, :b, :c, :b],
                            :upper_left_x     => [1., 3.5, 7., 10.],
                            :lower_right_x    => [2., 4.5, 8., 11.],
                            :upper_left_y     => [1., 3.5, 7., 10.],
                            :lower_right_y    => [2., 4.5, 8., 11.]
                      ] ) )

ods = ObjectDetectionStats( [ :a, :b, :c ]; IoU_threshold = 0.5 )

ods( KeyDF, GT )

ods.TP_or_FN.Highest_Prediction
ods.TP_or_FN.Intersection_Over_Union

ods.FP_or_TN.Highest_Prediction
ods.FP_or_TN.Intersection_Over_Union

ods.FN.Highest_Prediction
ods.FN.Intersection_Over_Union

function PR_curve( ods::ObjectDetectionStats, class_label::Symbol )

end

function average_precision( ods::ObjectDetectionStats, class_label::Symbol )
    max_TPs = first( size( ods.TP_or_FN ) )
    class_wise_TP_or_FN = filter( row -> row.class_labels != class_label, ods.TP_or_FN )
    class_wise_FP_or_FN = filter( row -> row.class_labels != class_label, ods.FP_or_FN )
    class_wise_FN       = filter( row -> row.class_labels != class_label, ods.FN )
    FNs = first( size( class_wise_FN ) )
    scores = vcat( class_wise_TP_or_FN.Highest_Prediction, class_wise_FP_or_FN.Highest_Prediction)
    scores = round.( scores, sigdigits = 3 )
    sort!( scores, rev = true )
    len_score = length( scores )
    statistics_df = DataFrame( Dict(    :Score => scores,
                                        :TP => zeros(Int, len_score),
                                        :FP => zeros(Int, len_score),
                                        :FN => repeat( [FNs], len_score),
                                        :Precision => zeros(Float64, len_score),
                                        :Recall => zeros(Float64, len_score)
                                ) )
    class_wise_TP_or_FN.Highest_Prediction .>
    for score in scores

    end
end
