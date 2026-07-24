
function flatten_dict(d::AbstractDict; sep::String = "__")
    flat = Dict{String, Any}()
    for (k, v) in d
        _flatten_into!(flat, string(k), v, sep)
    end
    return flat
end

# Decide whether a value should be treated as a "leaf" (kept as-is)
# or expanded further (dict / struct).
function _is_leaf(v)
    return v isa Union{AbstractString, Number, Symbol, Bool, Char,
                        Nothing, Missing, AbstractArray, Tuple, Function} ||
           v isa Type
end

_is_empty_struct(v) = isstructtype(typeof(v)) && fieldcount(typeof(v)) == 0

function _flatten_into!(flat::Dict{String, Any}, prefix::String, v, sep::String)
    if v isa AbstractDict
        for (k2, v2) in v
            _flatten_into!(flat, prefix * sep * string(k2), v2, sep)
        end
    elseif !_is_leaf(v) && _is_empty_struct(v)
        flat[prefix] = string(typeof(v))
    elseif !_is_leaf(v) && isstructtype(typeof(v))
        d2 = struct2dict(v)
        flat[prefix] = string(typeof(v))
        for (k2, v2) in d2
            _flatten_into!(flat, prefix * sep * string(k2), v2, sep)
        end
    elseif v isa AbstractVector || v isa Tuple
        flat[prefix] = Tuple(string(v2) for v2 in v)
    else
        flat[prefix] = v
    end
end
