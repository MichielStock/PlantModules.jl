@variables t, [description = "Time", unit = u"hr"]; #! add documentation
d = Differential(t);

"""
    hydraulic_module(; name, T, ρ_w, shape, Γ)

Returns a ModelingToolkit ODESystem describing the turgor-driven growth of a plant compartment.
WARNING: this module still requires an equation to be given for the osmotically active metabolite content M.
"""
function hydraulic_module(; name, T, ρ_w, shape::Shape, Γ, P, W, D)
    num_D = length(shape.ϵ_D)
    @constants (
        P_0 = 0.0, [description = "Minimum pressure", unit = u"MPa"],
        R = 8.314e-6, [description = "Ideal gas constant", unit = u"MJ / K / mol"]
    )
    @parameters (
        T = T, [description = "Temperature", unit = u"K"],
        ρ_w = ρ_w, [description = "Density of water", unit = u"g / m^3"],
        ϵ_D[1:num_D] = shape.ϵ_D, [description = "Dimensional elastic modulus", unit = u"MPa"],
        ϕ_D[1:num_D] = shape.ϕ_D, [description = "Dimensional extensibility", unit = u"MPa^-1 * hr^-1"],
        Γ = Γ, [description = "Critical turgor pressure", unit = u"MPa"]
    )
    @variables (
        Ψ(t), [description = "Total water potential", unit = u"MPa"],
        Π(t), [description = "Osmotic water potential", unit = u"MPa"],
        P(t) = P, [description = "Hydrostatic potential", unit = u"MPa"],
        M(t), [description = "Osmotically active metabolite content", unit = u"mol / m^3"], # m^3 so units match in second equation (Pa = J/m^3) #! extend validation function so L is ok?
        W(t) = W, [description = "Water content", unit = u"g"],
        D(t)[1:num_D] = D, [description = "Dimensions of compartment", unit = u"m"],
        V(t), [description = "Volume of compartment", unit = u"m^3"],
        ΣF(t), [description = "Net incoming water flux", unit = u"g / hr"],
        
        ΔP(t), [description = "Change in hydrostatic potential", unit = u"MPa / hr"],
        ΔW(t), [description = "Change in water content", unit = u"g / hr"],
        ΔD(t)[1:num_D], [description = "Change in dimensions of compartment", unit = u"m / hr"],
    )

    eqs = [
        Ψ ~ Π + P, # Water potential consists of a solute- and a pressure component
        Π ~ -R*T*M, # Solute component is determined by concentration of dissolved metabolites
        ΔW ~ ΣF, # Water content changes due to flux (depending on water potentials as defined in connections)
        V ~ W / ρ_w, # Shape is directly related to water content        
        V ~ volume(shape, D), # Shape is also directly related to compartment dimensions
        [ΔD[i] ~ D[i] * (ΔP/ϵ_D[i] + ϕ_D[i] * LSE(P - Γ, P_0, γ = 100)) for i in eachindex(D)]..., # Compartment dimensions can only change due to a change in pressure

        d(P) ~ ΔP,
        d(W) ~ ΔW,
        [d(D[i]) ~ ΔD[i] for i in eachindex(D)]...,
    ]
    return ODESystem(eqs, t; name)
end

"""
    constant_carbon_module(; name, C)

Returns a ModelingToolkit ODESystem describing a constant amount of osmotically active metabolite content.
"""
function constant_carbon_module(; name, M)
    # @parameters M = M [description = "Value of the constant amount of osmotically active metabolite content", unit = u"mol"]
    @variables ( #! change naming
        M(t) = M, [description = "Osmotically active metabolite content", unit = u"mol / m^3"],
        M_amount(t), [description = "Amount of osmotically active metabolite content", unit = u"mol"],
        V(t), [description = "Volume of compartment", unit = u"m^3"],
    )

    eqs = [
        M_amount ~ M*V,
        d(M_amount) ~ 0 #! need to rewrite as d(M) ~ ... so initial value of M gets used
    ]
    return ODESystem(eqs, t; name)
end

"""
    environmental_module(; name, T, ρ_w, W_max)

Returns a ModelingToolkit ODESystem describing a non-growing water reservoir.
WARNING: this module still requires an equation to be given for the total water potential Ψ.
"""
function environmental_module(; name, T, ρ_w, W_max, W_r)
    @parameters (
        T = T, [description = "Temperature", unit = u"K"],
        ρ_w = ρ_w, [description = "Density of water", unit = u"g / m^3"],
        W_max = W_max, [description = "Water capacity of compartment", unit = u"g"],
        )
    @variables (
        Ψ(t), [description = "Total water potential", unit = u"MPa"],
        W(t) = W_r * W_max, [description = "Water content", unit = u"g"],
        W_r(t), [description = "Relative water content", unit = u"g / g"],
        ΣF(t), [description = "Net incoming water flux", unit = u"g / hr"],

        ΔW(t), [description = "Change in water content", unit = u"g / hr"],
    )

    eqs = [
        W_r ~ W / W_max,
        ΔW ~ ΣF, # Water content changes due to flux (depending on water potentials as defined in connections)
        d(W) ~ ΔW,

        # prevent variables from being erased from existence
        Ψ ~ Ψ,
        T ~ T
    ]
    return ODESystem(eqs, t; name)
end

# Module connections #

"""
    hydraulic_connection(; name, K)

Returns a ModelingToolkit ODESystem describing a water flow connection between two hydraulics-based functional modules. 
"""
function hydraulic_connection(; name, K)
    @parameters (
        K = K, [description = "Hydraulic conductivity of connection", unit = u"g / hr / MPa"],
    )
    @variables (
        F(t), [description = "Water flux from compartment 2 to compartment 1", unit = u"g / hr"],
        Ψ_1(t), [description = "Total water potential of compartment 1", unit = u"MPa"],
        Ψ_2(t), [description = "Total water potential of compartment 2", unit = u"MPa"],
    )

    eqs = [
        F ~ K * (Ψ_2 - Ψ_1)
    ]
    return ODESystem(eqs, t; name)
end

hydraulic_connection_eqs(node_MTK, nb_node_MTKs, connection_MTKs) = [
	[connection_MTK.Ψ_1 ~ node_MTK.Ψ for connection_MTK in connection_MTKs]...,
	[connection_MTK.Ψ_2 ~ nb_node_MTK.Ψ for (connection_MTK, nb_node_MTK) in zip(connection_MTKs, nb_node_MTKs)]...,
	node_MTK.ΣF ~ sum([connection_MTK.F for connection_MTK in connection_MTKs])
] #! put this in func_modules and explain to user they gotta provide this stuff mr white yo

# Helper functions #

## Unitful is a dangerous beast
val(x) = x
val(x::Quantity) = x.val

## Forbidden rites #! try fixing this... creative solution?
import Base.exp
import Base.log
exp(x::Quantity) = exp(val(x))*unit(x)
log(x::Quantity) = log(val(x))*unit(x)

"""
    LSE(x, y, ...; γ = 1)

LogSumExp, a smooth approximation for the maximum function. 
The temperature parameter γ determines how close the approximation is to the actual maximum.
This implementation makes use of the log-sum-exp trick (https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations) to ensure numerical stability.
"""
LSE(x::Real...; γ = 1) = log(sum(exp.(γ .* x .- maximum(x))) ) / γ + maximum(x)
@register_symbolic LSE(x)

# Default values #

default_params = (
    hydraulic_module = (
        T = 298.15, ρ_w = 1.0e6, shape = Sphere(ϵ_D = [1.0], ϕ_D = [1.0]), Γ = 0.3
    ),
    constant_carbon_module = (
    ),
    environmental_module = (
        T = 298.15, ρ_w = 1.0e6, W_max = 1e6
    ),
    hydraulic_connection = (
        K = 600,
    )
)

default_u0s = (
    hydraulic_module = (
        P = 0.1, D = [0.1], W = volume(Sphere(ϵ_D = [1.0], ϕ_D = [1.0]), [0.1]) * 1.0e6,
    ),
    constant_carbon_module = (
        M = 1000,
    ),
    environmental_module = (
        W_r = 0.6,
    ), 
    hydraulic_connection = (
    )
)