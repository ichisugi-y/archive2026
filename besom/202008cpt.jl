#=
//Copyright (c) 2020 National Institute of Advanced Industrial Science and Technology (AIST), All Rights Reserved.
//Author: Yuuji Ichisugi

ＢＥＳＯＭ４ 条件付確率モデル

## 実行方法

### パターンマッチのテスト
	julia> include("cpt.jl")
	julia> testAll()



◆実装メモ
・ Flux.jl のバグを回避するため重みの保持には多次元配列を使う。
異なるサイズのノードの混在はあきらめる。

・内包表現ではなく Zygote.Buffer を使う。読みやすさ優先。

・ノードが値 i ∈ {1,2,...,n, φ} を値に取るとき、
ユニットのφ値はインデックス end 、値 i はインデックス i を使って表現。


=#




# mutable struct Node
#     parents::Any     # parent nodes
#     children::Any     # child nodes, currently not used
#     size::Int     # number of units in this node
#     Node(size) = (n = new(); n.size = size; n)
# end

abstract type NodeParam end

mutable struct CPT4v2 <: NodeParam
    uNodes::Int
    uUnits::Int
    dNodes::Int
    dUnits::Int
    b_ciud::Array{Float32,4}
    b_ujdk::Array{Float32,4}
    CPT4v2(uNodes, uUnits, dNodes, dUnits) = begin
        b_ciud = fill(-Inf32, uNodes, uUnits, uNodes, dNodes)
        b_ujdk = fill(-Inf32, uNodes, uUnits, dNodes, dUnits)
        new(uNodes, uUnits, dNodes, dUnits, b_ciud, b_ujdk)
    end
end

function printRawParams(p::CPT4v2)
    for c in 1:p.uNodes, i in 1:p.uUnits, u in 1:p.uNodes
        print("w_ciud[$c,$i,$u,:]={ ")
        for d in 1:p.dNodes
            print("$(param.b_ciud[c,i,u,d]), ")
        end
        println("}")
    end
    for u in 1:p.uNodes, j in 1:p.uUnits, d in 1:dNodes
        print("w_ujdk[$u,$j,$d,:]={ ")
        for k in 1:p.dUnits
            print("$(param.b_ujdk[u,j,d,k]), ")
        end
        println("}")
    end
end


# 配列の配列・・・の形で Julia 文法のまま出力。
function printParams(p::CPT4v2)
    #println("w_ciud = [")               # XXX: 出力されない。なぜ？？
    println("w_ciud = [")
    for c in 1:p.uNodes
        println("  [")
        for i in 1:p.uUnits
            print("    [")
            for u in 1:p.uNodes
                print("[")
                for d in 1:p.dNodes
                    print("$(sigmoid(p.b_ciud[c,i,u,d])), ")
                end
                print("], ")
            end
            println("],")
        end
        println("  ],")
    end
    println("]")
    println("w_ujdk = [")
    for u in 1:p.uNodes
        println("  [")
        for j in 1:p.uUnits
            print("    [")
            for d in 1:p.dNodes
                print("[")
                for k in 1:p.dUnits
                    print("$(sigmoid(p.b_ujdk[u,j,d,k])), ")
                end
                print("], ")
            end
            println("],")
        end
        println("  ],")
    end
    println("]")
end



##--------------------------------------------------

using Zygote: Buffer

# (-∞, ∞) => (0,1)
function sigmoid(x::Float32)
    1 / (1 + exp(-x))
end

# (0,1) => (-∞, ∞)
# Inverse function of sigmoid.
function logit(y)
    log(y / (1 - y))
end

#=
上位層にノードに与えた one hot vector 表現による値をもとに
下位層のノードの周辺確率分布を計算。
=#
function calcCPs(p::CPT4v2, pvals::Array{Array{Float32,1},1})
    @assert length(pvals) == p.uNodes

    w_ciud = sigmoid.(p.b_ciud)
    w_ujdk = sigmoid.(p.b_ujdk)

    g_ud = Buffer(zeros(Float32, p.uNodes, p.dNodes))
    for u in 1:p.uNodes, d in 1:p.dNodes
        val = 1f0
        for c in 1:p.uNodes, i in 1:p.uUnits
            val *= 1 - w_ciud[c,i,u,d] * pvals[c][i]
        end
        g_ud[u,d] = val
    end

    s_dk = Buffer(zeros(Float32, p.dNodes, p.dUnits))
    for d in 1:p.dNodes, k in 1:p.dUnits
        val = 0f0
        for u in 1:p.uNodes, j in 1:p.uUnits
            val += w_ujdk[u,j,d,k] * g_ud[u,d] * pvals[u][j]
        end
        s_dk[d,k] = val
    end

    function x_dk(d)
        x_k = Buffer(zeros(Float32, p.dUnits + 1))
        for i in 1:p.dUnits
            x_k[i] = s_dk[d,i]
        end
        x_k[length(x_k)] = 0f0  # 初期化が必要 → 不要？

        epsilon = 1f-6      # 1 x 10^(-6) as Float32
        Z = max(1f0, sum(s_dk[d,:])) + epsilon
        for i in 1:p.dUnits
            x_k[i] = s_dk[d,i] / Z
        end
        last = length(x_k)
        x_k[last] = 1 - sum(x_k[1:last-1])    # Buffer does not support "end".
        copy(x_k)
    end
    ret = [x_dk(d) for d in 1:p.dNodes]
    return ret
end


nodeActivationRate = 0.1f0
#nodeActivationRate = 0.75f0
nodeSmoothingCoefficient = 0.001f0

# 上位層のノードの事前分布を配列の配列の形で返す。
function calcRootNodesP(p::CPT4v2)
    function x_uj()
        x_j = Buffer(zeros(Float32, p.uUnits + 1))
        a = nodeActivationRate / p.uUnits
        for j in 1:p.uUnits
            x_j[j] = a
        end
        x_j[length(x_j)] = 1 - nodeActivationRate
        copy(x_j)
    end
    return [x_uj() for u in 1:p.uNodes]
end


#=
・すべてのノードに与えた値をもとに同時確率を計算。
変数の値は one hot vector で与える。
=#
function calcJointP(p::CPT4v2, vals::Array{Array{Float32,1},1})
    #@show vals
    @assert p.uNodes + p.dNodes == length(vals)
    pvals = vals[1:p.uNodes]
    cvals = vals[p.uNodes+1:end]
    ret = 1f0
    # root node の事前分布を計算
    x_uj = calcRootNodesP(p)
    for u in 1:p.uNodes
        # one hot vector pvals との内積を計算
        sum = 0f0
        for j = 1:p.uUnits+1
            sum += x_uj[u][j] * pvals[u][j]
        end
        ret *= sum
    end
    #@show x_uj
    #@show pvals
    #@show ret
    x_dk = calcCPs(p, pvals)
    for d in 1:p.dNodes
        # one hot vector cvals との内積を計算
        sum = 0f0
        for k = 1:p.dUnits+1
            sum += x_dk[d][k] * cvals[d][k]
        end
        ret *= sum
    end
    #@show x_dk
    #@show cvals
    #@show ret
    return ret
end


function setOneHot(a::Array{Float32,2}, i, v)
    n,u = size(a)
    for j = 1:u
        a[i,j] = 0
    end
    a[i,v] = 1
end
function setOneHot(a::Array{Float32,2}, v::Array{Int64,1})
    n,u = size(a)
    for i = 1:n
        setOneHot(a, i, v[i])
    end
end



##


function valueCombination(nodes, units)
    valueCombination1(nodes, units, 1)
end
function valueCombination1(nodes, units, n)
    if n == nodes
        ret = Array{Array{Float32,1},1}[]
        for j = 1:units + 1
            push!(ret, [oneHotVec(units + 1, j)])
        end
        return ret
    else
        tails = valueCombination1(nodes, units, n + 1)
        ret = Array{Array{Float32,1},1}[]
        for j = 1:units + 1
            head = oneHotVec(units + 1, j)
            for tail in tails
                push!(ret, vcat([head], tail))
            end
        end
        return ret
    end
end
function oneHotVec(n, i)
    ret = zeros(Float32, n)
    if i == 0
        ret[end] = 1f0
    else
        ret[i] = 1f0
    end
    return ret
end


##--------------------------------------------------

using Printf

# 上位層のすべての値の組み合わせにとそれに伴う子ノードの値の事後分布を print する。
function printCPT(p::CPT4v2)
    #@show bnet.parents
    #@show valueCombination(bnet.parents)
    for pvals in valueCombination(p.uNodes, p.uUnits)
        for pval in pvals
            print(Int.(pval))
        end
        print("->")
        for cval in calcCPs(p, pvals)
            print("[")
            for val in cval
                @printf("%.1f, ", val)
            end
            print("]")
        end
        println()
    end
end

# デバッグ用。同時確率をソートして出力。
function printRawJPlist(p::CPT4v2, cvals::Array{Array{Float32,1},1})
    uppers = valueCombination(p.uNodes, p.uUnits)
    pairs = [pvals=>calcJointP(p, vcat(pvals,cvals))
             for pvals in uppers]
    println("Result for input $cvals:")
    for pair in sort(pairs, by=(pair)->pair.second)
        @printf("%s : %f (%e)", pair.first, pair.second, pair.second)
        println()
    end
    println()
end


# Print sorted list of pairs of parent values and joint porabilities.
# 出力を読みやすく整形。論文に合わせてφ値ユニットは０番目の要素に。
function printJPlist(p::CPT4v2, cvals::Array{Array{Float32,1},1})
    uppers = valueCombination(p.uNodes, p.uUnits)
    pairs = [pvals=>calcJointP(p, vcat(pvals,cvals))
             for pvals in uppers]
    println("Result for input $cvals:")
    for pair in sort(pairs, by=(pair)->pair.second, rev=true)
        print("(")
        @assert length(pair.first) >= 1
        print(oneColdZeroOrigin(pair.first[1]))
        for i in 2:length(pair.first)
            print(", ")
            print(oneColdZeroOrigin(pair.first[i]))
        end
        print(") : JP=")
        @printf("%f", pair.second)
        println()
        #@printf("%s : JP=%f (%e)\n", pair.first, pair.second, pair.second)
    end
    println()
end
function oneColdZeroOrigin(val::Array{Float32,1})
    i = argmax(val)
    return i == length(val) ? 0 : i
end


# 論文用に出力をより読みやすく整形。vNodes=2, vUnits=3 で固定。
function matchResults(expr, ivals::Array{Int64,1})
    @show expr, ivals
    vNodes=2
    vUnits=3
    cvals = [oneHotVec(vUnits+1, i) for i in vcat(ivals, 1)]
    p = defPattern(expr, vNodes=vNodes, vUnits=vUnits)
    uppers = valueCombination(p.uNodes, p.uUnits)
    pairs = [pvals=>calcJointP(p, vcat(pvals,cvals))
             for pvals in uppers]
    println("Sorted results:")
    for pair in sort(pairs, by=(pair)->pair.second, rev=true)
        patternID = oneColdZeroOrigin(pair.first[1])
        if patternID == 0
            @printf("P=0, ")
        else
            @printf("P=\"%s\", ", expr.args[patternID])
        end
        @assert vNodes == 2
        @printf("X=%s, ", oneColdZeroOrigin(pair.first[2]))
        @printf("Y=%s", oneColdZeroOrigin(pair.first[3]))
        print(" : JP=")
        @printf("%f", pair.second)
        println()
        #@printf("%s : JP=%f (%e)\n", pair.first, pair.second, pair.second)
    end
    println()
end


## MPE
function calcMPE(p::CPT4v2, cvals::Array{Array{Float32,1},1})
    ##### XXX: 実装中＊＊＊＊＊＊＊＊＊
    uppers = valueCombination(p.uNodes, uUnits)
    maxP = -1f0
    mpe = nothing
    for nvecs in uppers
        vals = vcat(pcals, cvals)
        p = calcJointP(bnet, vals)
        if p > maxP
            maxP = p
            mpe = vals
        end
    end
    return mpe
end

#=
上の層の変数のすべての値の組み合わせから生成される
下の層の値のベクトルを返す。学習データとして使われる。
微分可能な実装でなくてもよい。
（サンプリング？？？）
=#
function makeSamples(p::CPT4v2)
    x_uj = fill(0f0, l.us, l.size + 1)
    ##### XXX: 実装中＊＊＊＊＊＊＊＊＊
    ret = []
    for u in l.us
        makeSamples1(l, x_uj)
        for j in 1:l.size
            vecs[i] .= nvecs[i]     # copy to x_ci, x_uj
        end
        x_dk = calcCPs(l, x_ci, x_uj)
        # x_dk は one hot でないかもしれないがそのまま配列にする。
        push!(ret, x_dk)
    end
    return ret
end


# 下の層のすべての値の組み合わせに対するＭＰＥを計算し print する。
function testMPE(p::CPT4v2)
    vec = makeSamples(l)
    for x_dk in vec
        println(x_dk, calcMPE(l, x_dk))
    end
end


##--------------------------------------------------


## Make a network for test
function makeSimpleNet()
    # ネットワーク構築
    cNodes = 1
    #cUnits = 3
    uNodes = 2
    uUnits = 3
    dNodes = 1
    dUnits = 3
#    cnode = Node(cUnits)
#    parents = vcat(cnode, [Node(uUnits) for i in 1:uNodes])
#    children = [Node(dUnits) for i in 1:dNodes]
#    bnet = BNet2(parents, children)
    p = CPT4v2(cNodes+uNodes, uUnits, dNodes, dUnits)

    printParams(p)

    # パラメタ設定
    for c = 1:cNodes, i = 1:uUnits, u = 1:uNodes, d = 1:dNodes
        p.b_ciud[c, i, cNodes+u, d] =
            (i == 1 && u == 1 ||
             i == 2 && u == 2 ||
             i == 3) ? Inf32 : -Inf32
    end
    for u = 1:uNodes, j = 1:uUnits, d = 1:dNodes, k = 1:dUnits
        p.b_ujdk[cNodes+u, j, d, k] =
            j == k ? Inf32 : -Inf32
    end
    return p
end



## Make a pattern matching network
#= テストに使うパターン：
(値からパターンが一意に決まるような簡単なタスクにする。)
[X, 1, 2]
[2, X, 1]
[X, 3, X]
=#
# 重みを設定。
function makePatternNet()
    cNodes = 1
    vNodes = 1
    vUnits = 3
    dNodes = 3
    dUnits = 3
    @assert vUnits == dUnits

    p = CPT4v2(cNodes+vNodes, vUnits, dNodes, dUnits)

    printParams(p)

    w_ciud = [
        [
            [[0,0,0], [0,1,1]],
            [[0,0,0], [1,0,1]],
            #[[0,0,0], [1,1,0]],
            [[0,0,0], [0,1,0]],
        ], [
            [[0,0,0], [0,0,0]],
            [[0,0,0], [0,0,0]],
            [[0,0,0], [0,0,0]],
        ]
    ]
    w_ujdk = [
        [
            [[0,0,0], [1,0,0], [0,1,0]],
            [[0,1,0], [0,0,0], [1,0,0]],
            [[0,0,0], [0,0,1], [0,0,0]],
        ], [
            [[1,0,0], [1,0,0], [1,0,0]],
            [[0,1,0], [0,1,0], [0,1,0]],
            [[0,0,1], [0,0,1], [0,0,1]],
        ],
    ]
    #    function toInf(x)
    #        return x == 1 ? Inf32 : -Inf32
    #    end
    #
    for c in 1:cNodes+vNodes,
        i in 1:vUnits,
        u in 1:cNodes+vNodes,
        d in 1:dNodes
        p.b_ciud[c,i,u,d] = logit(w_ciud[c][i][u][d])
    end
    for u in 1:cNodes+vNodes,
        j in 1:vUnits,
        d in 1:dNodes,
        k in 1:dUnits
        p.b_ujdk[u,j,d,k] = logit(w_ujdk[u][j][d][k])
    end

    printParams(p)
    return p
end

#=
An example of expr:

julia> dump(:[
       [x,1]=>1.2
       [x,x]=>2.3
       ])
Expr
  head: Symbol vcat
  args: Array{Any}((2,))
    1: Expr
      head: Symbol call
      args: Array{Any}((3,))
        1: Symbol =>
        2: Expr
          head: Symbol vect
          args: Array{Any}((2,))
            1: Symbol x
            2: Int64 1
        3: Float64 1.2
    2: Expr
      head: Symbol call
      args: Array{Any}((3,))
        1: Symbol =>
        2: Expr
          head: Symbol vect
          args: Array{Any}((2,))
            1: Symbol x
            2: Symbol x
        3: Float64 2.3

=#
function defPattern(expr::Expr; vNodes=3, vUnits=3)
    # 複数行に書けば vcat 、要素が１つなら vect 、１行で書けば hcat
    # コンマで区切れば１行でも複数行でも vect
    @assert expr.head == :vcat || expr.head == :vect || expr.head == :hcat
    # size of input vector
    pairs = expr.args
    dNodes = length(pairs[1].args[2].args)
    vUnits == length(pairs) || error("the number of pattenrs must be $vUnits")
    for pair in pairs
        length(pair.args[2].args) == dNodes || error("length of $pair is not $dNodes")
    end
    p =  CPT4v2(vNodes+1, vUnits, dNodes+1, vUnits)
    # パラメタ設定
    vars = [:X, :Y, :Z]
    wildcard = :_
    @assert vNodes <= length(vars)

    # var nodes と input nodes の間の結合。それぞれ対角線状に結合
    for u in 1:vNodes, j in 1:vUnits, d in 1:dNodes
        p.b_ujdk[1+u,j,d,j] = Inf32
    end

    for (i,pair) in enumerate(pairs)
        elems = pair.args[2].args
        value = pair.args[3]
        0 <= value <= 1 || error("value $value is not in [0,1]")
        function closeVariableGates(i,d)
            for u in 1:vNodes
                p.b_ciud[1,i,1+u,d] = Inf32
            end
        end
        for (d,e) in enumerate(elems)
            if e == 0
                closeVariableGates(i,d)
            elseif e == wildcard
                closeVariableGates(i,d)
                b = logit(1f0/(vUnits+1))
                for k in 1:vUnits
                    p.b_ujdk[1,i,d,k] = b
                end
            elseif e == :+
                closeVariableGates(i,d)
                b = logit(1f0/(vUnits))
                for k in 1:vUnits
                    p.b_ujdk[1,i,d,k] = b
                end
            elseif typeof(e) <: Integer
                1 <= e <= vUnits || error("number out of range: $e")
                closeVariableGates(i,d)
                p.b_ujdk[1,i,d,e] = Inf32
            elseif typeof(e) == Symbol
                for u in 1:vNodes
                    if vars[u] != e
                        p.b_ciud[1,i,1+u,d] = Inf32
                    end
                end
            else
                error("illegal pattern element: $e")
            end
        end
        # set the weight of the value node
        @assert typeof(value) <: Real
        p.b_ujdk[1,i,end,1] = logit(Float32(value))
    end

    #printParams(p)

    return p
end

function testPatternNet1()
    defPattern(
        :[
            [X, 1, 2]=>1
            [2, X, 1]=>1
            [1, 2, X]=>1
            [X, 3, X]=>1
        ]
    )
end



##--------------------------------------------------
## test

function test1(bnet)
    @show bnet
    printParams(bnet)
    printCPT(bnet)
    #@show makeSamples(bnet)
    #testMPE(bnet)
end

function testAll()
    test1(makeSimpleNet())
    test1(makePatternNet())
    #test1(testPatternNet1()) # defPattern test

    pat = defPattern(:[
        [X, 1, 2]=>1
        [X, X, 2]=>1
        [1, X, Y]=>1
    ], vNodes=2)
    printJPlist(pat, [[1f0,0,0,0],[1f0,0,0,0],[0,1f0,0,0],[1f0,0,0,0]]) # [1,1,2]
    printJPlist(pat, [[1f0,0,0,0],[0,1f0,0,0],[0,1f0,0,0],[1f0,0,0,0]]) # [1,2,2]

    pat = defPattern(:[
        [X, 1, 2]=>0.1
        [X, X, 2]=>0.2
        [1, X, Y]=>1
    ], vNodes=2)
    printJPlist(pat, [[1f0,0,0,0],[1f0,0,0,0],[0,1f0,0,0],[1f0,0,0,0]]) # [1,1,2]


    pat = defPattern(:[
        [0, 1, 2]=>1
        [+, 1, 2]=>1
        [_, 1, 2]=>1
        #    [X, 1, 2]=>1
        #    [X, X, 2]=>1
        #    [X, Y, 2]=>1
        #    [1, 1, 2]=>1
    ], vNodes=2)
    printJPlist(pat, [[0,0,0,1f0],[1f0,0,0,0],[0,1f0,0,0],[1f0,0,0,0]])# [0,1,2]
    printJPlist(pat, [[1f0,0,0,0],[1f0,0,0,0],[0,1f0,0,0],[1f0,0,0,0]])# [1,1,2]
    printJPlist(pat, [[0,1f0,0,0],[1f0,0,0,0],[0,1f0,0,0],[1f0,0,0,0]])# [2,1,2]

    pat = defPattern(:[
        [_, _, 2]=>1
        [X, X, 2]=>1
        [X, Y, 2]=>1
    ], vNodes=2)
    printJPlist(pat, [[1f0,0,0,0],[1f0,0,0,0],[0,1f0,0,0],[1f0,0,0,0]])# [1,1,2]
    printJPlist(pat, [[1f0,0,0,0],[0,1f0,0,0],[0,1f0,0,0],[1f0,0,0,0]])# [1,2,2]
end     # function testAll()

#testAll()

# pat = makePatternNet()
# printJPlist(pat, [[1f0,0,0,0],[1f0,0,0,0],[0,1f0,0,0]])# [1,1,2]
# printJPlist(pat, [[0,1f0,0,0],[0,0,1f0,0],[0,1f0,0,0]])# [2,3,2]

#--------------------------------------------------
# 論文用デモ

# マッチするパターンが１つだけの例
function matchDemoAll()
    matchResults(:[
        [X, 1, 2]=>1
        [X, X, 2]=>1
        [X, Y, 2]=>1
    ], [1,2,2])

    # 最も特殊なパターンが選択される例
    matchResults(:[
        [X, 1, 2]=>1
        [X, X, 2]=>1
        [X, Y, 2]=>1
    ], [1,1,2])

    # 最も特殊なパターンの中で最も価値が高いものが選択される例
    matchResults(:[
        [X, 1, 2]=>0.2
        [X, X, 2]=>0.1
        [X, Y, 2]=>1
    ], [1,1,2])

    # 価値の差がありすぎて逆転する例
    matchResults(:[
        [X, 1, 2]=>0.02
        [X, X, 2]=>0.01
        [X, Y, 2]=>1
    ], [1,1,2])

    # ワイルドカードの使用例１
    matchResults(:[
        [_, 1, 2]=>1
        [X, X, 2]=>1
        [_, _, 2]=>1
    ], [1,1,2])

    # ワイルドカードの使用例２
    matchResults(:[
        [_, 1, 2]=>1
        [X, X, 2]=>1
        [_, _, 2]=>1
    ], [3,1,2])

    # 値０の扱い
    matchResults(:[
        [0, 1, 2]=>1
        [X, 1, 2]=>1
        [2, 1, 2]=>1
    ], [0,1,2])

    # パターン＋の使用例
    matchResults(:[
        [+, 1, 2]=>1
        [0, 1, 2]=>1
        [2, 1, 2]=>1
    ], [0,1,2])

    # パターン＋の使用例
    matchResults(:[
        [+, 1, 2]=>1
        [0, 1, 2]=>1
        [2, 1, 2]=>1
    ], [1,1,2])
end

#matchDemoAll()
