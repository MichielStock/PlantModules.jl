using Pkg; Pkg.activate(".")

using ModelingToolkit, Plots, DifferentialEquations, Unitful

abstract type Volume end

struct Sphere<:Volume
    ζ::Vector
    formula::Function
    function Sphere(; ζ::Vector)
        length(ζ) != 1 && error("An array of length $(length(ζ)) was given for ζ while length 1 was expected.")

        return new(ζ, D -> 4/3 * pi * D[1]^3)
    end
end

@variables t, [description = "Time", unit = u"s"];
d = Differential(t);

fruitvolume = Sphere(ζ = [1e-5])

@constants (
R = 8.314, [description = "Ideal gas constant", unit = u"J / K / mol"],
)

@parameters (
    ζ[1:length(fruitvolume.ζ)] = fruitvolume.ζ, [unit = u"m / m / Pa / s"], # extensibility of sides
    P_ext = 1.0e5, [unit = u"Pa"], # exterior pressure on volume
    T = 293.15, [unit = u"K"], # temperature
    n = 1, [unit = u"mol"], # moles of gas inside volume
)

@variables(
    D(t)[1:length(fruitvolume.ζ)], [unit = u"m"], # dimensions of volume
    P(t), [unit = u"Pa"], # pressure inside volume
    V(t), [unit = u"m^3"] # volume of volume
)

eqs = [
    (d.(D) ./ D .~ ζ * (P - P_ext))...,
    V ~ fruitvolume.formula(D),
    P ~ n*R*T/V,
]


@named sys = ODESystem(eqs, t)

u0 = [
    D[1] => 0.1
]


sys_simp = structural_simplify(sys)

u0_req = ModelingToolkit.missing_variable_defaults(sys_simp) # generates zero initial value for ALL initial states
u0_ext = union(u0, u0_req) # add actual values of non-dummy initial states 
u0_full = unique(x -> x[1], u0_ext) # remove duplicates

prob = ODEProblem(sys_simp, u0_full, (0, 10))
sol = solve(prob)
plots = [plot(sol, idxs = [var]) for var in [D[1], P, V]]
plot(plots..., layout = (3, 1))

using BenchmarkTools

@btime solve(prob)