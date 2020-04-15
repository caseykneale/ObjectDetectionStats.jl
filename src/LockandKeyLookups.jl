using DataFrames

mutable struct LockandKeyLookup{A,B,C}
    key::Union{ AbstractDataFrame, AbstractArray }
    tumbler::Union{ Vector, Tuple }
    key_lookup_fn::A#Function
    pin_lookup_fn::B#Function
    emission_fn::C#Function
    key_length::Int
    tumblers::Vector{ Int }
end

Base.length(lkl::LockandKeyLookup)::Int = lkl.key_length

"""
    LockandKeyLookup(   key, tumbler,
                        key_lookup_fn, pin_lookup_fn,
                        emitter_fn = ( k, t ) -> key_lookup_fn( k ) == pin_lookup_fn( t ) )

Instantiates a `LockandKeyLookup` iterable object.
The `key` is the iterator to find a matching condition for across the `tumbler` iterators or internally called "pins".

the `_lookup_fn` functions are the functions used to `select` the data from each iteration for matching.
Ie: `key_lookup_fn`     = row -> row[!, [ :id, :time, :address ] ]
    `tumbler_lookup_fn` = row -> row[!, [ :row_id, :t, :Address ] ]

The `emitter_fn` is the function used to determine if there is infact a match, by default it asseses if the lookup functions equate.

"""
function LockandKeyLookup(  key, tumbler,
                            key_lookup_fn, pin_lookup_fn,
                            emitter_fn = ( k, t ) -> key_lookup_fn( k ) == pin_lookup_fn( t ) )
                            #Base.:(==) )#
    return LockandKeyLookup(    key, tumbler,
                                key_lookup_fn, pin_lookup_fn, emitter_fn,
                                first( size( key ) ), length.(tumbler) )
end


function get_smallest_pin( tumbler_values, tumbler_states, fn::Function )::Int
    not_nothing     = .!isnothing.( last.( tumbler_states ) )
    is_something    = sum( not_nothing )
    if is_something == 0
        return 0
    elseif is_something == 1
        return findfirst( not_nothing )
    elseif is_something > 1
        smallest_pin = argmin( fn.( tumbler_values[ not_nothing ] ) )
        return findall( not_nothing )[smallest_pin]
    end
end

function Base.iterate( lkl::LockandKeyLookup, state = ( iterate( lkl.key ), iterate.(lkl.tumbler) ) )
    ( key_value, key_state ), tumbler_values_and_states = state
    ( tumbler_values, tumbler_states ) = first.(tumbler_values_and_states), last.(tumbler_values_and_states)

    (last( key_state )  > lkl.key_length) && return nothing
    get_key             = lkl.key_lookup_fn( key_value )
    smallest_pin        = get_smallest_pin( tumbler_values, tumbler_states, lkl.pin_lookup_fn )
    (smallest_pin == 0) && return nothing #?
    #if the key is < then the lowest pin then roll the key
    if get_key < lkl.pin_lookup_fn( tumbler_values[smallest_pin] )
        return  ( key_state => (0, 0) ), #no match for key
                ( Base.iterate(lkl.key, key_state ), (tumbler_values_and_states) )
    else #otherwise roll the tumbler until there's a match or there's nothing left.
        while true
            if lkl.emission_fn( key_value, tumbler_values[ smallest_pin ] )
                break
            end
            newtumbler = Base.iterate( lkl.tumbler[ smallest_pin ], tumbler_states[ smallest_pin ] )
            isnothing(newtumbler) && return nothing
            tumbler_values[ smallest_pin ] = first( newtumbler )
            tumbler_states[ smallest_pin ] = last( newtumbler )
            smallest_pin        = get_smallest_pin( tumbler_values, tumbler_states, lkl.pin_lookup_fn )
        end
        return  ( last(key_state) => ( smallest_pin, last(tumbler_states[ smallest_pin ]) ) ),
                ( Base.iterate( lkl.key, key_state ), zip( tumbler_values, tumbler_states ) )
    end
end

#examples...
a = DataFrame( Dict( :a => 1:1:110, :b => 1:1:110 ) )
b = DataFrame( Dict( :b => 6:2:120, :c => 6:2:120 ) )
c = DataFrame( Dict( :b => 7:2:90, :c => 7:2:90 ) )

for i in LockandKeyLookup( eachrow(a), eachrow.([b, c]), X -> X.a, X -> X.b )
    println( i )
end

using BenchmarkTools
function proposed(a::DataFrame, b::DataFrame, c::DataFrame)
    return  [ i for i in LockandKeyLookup( eachrow(a), eachrow.([b, c]), X -> X.a, X -> X.b ) ]
end
@benchmark proposed(a, b, c)

#alternative
function alternative(a::DataFrame, b::DataFrame, c::DataFrame)::DataFrame
    x = join(a, sort( vcat( b, c ), :b ), on = :b, kind = :left)
    return x
end
@benchmark alternative(a, b, c)

a = DataFrame( Dict( :a => 1:10, :b => 1:10 ) )
iterator = eachrow(a)
iterate(last(iterate( iterator )))
