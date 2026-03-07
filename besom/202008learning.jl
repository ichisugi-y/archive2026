#=
//Copyright (c) 2020 National Institute of Advanced Industrial Science and Technology (AIST), All Rights Reserved.
//Author: Yuuji Ichisugi

ＢＥＳＯＭ４ ＣＰＴモデル
学習

## 実行方法

### 学習のテスト（1000 エポック）
	julia> include("learning.jl")

### 訓練データを間引いた学習のテスト（1000 エポック）
        julia> learningTest4skip()

=#

include("cpt.jl")


using Flux
using Flux: @epochs, mse, train!
using Flux: throttle
#using Plots


#=
 上位層のすべての値の組み合わせごとに子ノードのＭＡＰ値を計算し、
値を並べたものの配列を学習データとする。
=#
function makeCompleteData(p::CPT4v2)
    valsList = []
    for pvals in valueCombination(p.uNodes, p.uUnits)
        # Assumes the return value of calcCPs is a one-hot vector .
        cvals = calcCPs(p, pvals)
        push!(valsList, vcat(pvals, cvals))
    end
    return valsList
end

#=
上の層の変数のすべての値の組み合わせから生成される
下の層の値のベクトルを返す。
=#
function makeInputData(p::CPT4v2)
    ret = Array{Array{Float32,1},1}[]
    for pvals in valueCombination(p.uNodes, p.uUnits)
        # x_dk は one hot でないかもしれないがそのまま配列にする。
        push!(ret, calcCPs(p, pvals))
    end
    return ret
end


# 重みをランダムに設定。 w は 0.5 付近の乱数値。（b では 0 付近）
function randParamValue(mean, range)
    # rand() は [0,1) の値。
    return (mean - range) + rand() * range * 2
end

function setRandomParams(p::CPT4v2, f)
    #setRandomParams(p.b_ciud, f)
    #setRandomParams(p.b_ujdk, f)
    @. p.b_ciud = f()
    @. p.b_ujdk = f()
end

# パターンノードの値が０である事前確率は０なので、
# 学習データからはそのようなデータは外す。
# （本当は変数ノードの値も事前確率に応じた頻度で訓練データに
# 含まれていなければならないが、いまは一様にしている。）
function defaultCompleteDataFilter(data)
    ret = []
    for d in data
        if oneColdZeroOrigin(d[1]) != 0
            push!(ret, d)
        end
    end
    return ret
end

# 完全データのリストから n 個のうち１つを周期的に間引く。
function skipSomeDataFilter(data; cycle=3)
    ddata = defaultCompleteDataFilter(data)
    ret = []
    c = 0
    for d in ddata
        c = c + 1
        if c == cycle
            c = 0
        else
            push!(ret, d)
        end
    end
    return ret
end



function defaultEnv()
    env = Dict()
    η = 0.2
    opt = Descent(η)
#opt = Flux.Optimiser(WeightDecay(0.5 * η), Descent(η))
#opt = Flux.Optimiser(WeightDecay(0.01 * η), Descent(η)) # 弱い
#opt = Flux.Optimiser(WeightDecay(0.99), Descent(η)) # 強い
#opt = Momentum()
opt = Momentum(0.01, 0.99)
    #opt = ADAM()
    env[:completeDataFilter] = defaultCompleteDataFilter
    env[:opt] = opt
    env[:loss] = likelihood
    env[:hloss] = hloss     # loss for trainWithHiddenLayer
    #env[:epochs] = 100
    env[:epochs] = 1000
    #env[:randParamFunc] = ()->randParamValue(0, 0.1)
    #env[:randParamFunc] = ()->randParamValue(-5, 0.5)
    #env[:randParamFunc] = ()->randParamValue(0, 0.5)
    env[:randParamFunc] = ()->randParamValue(0, 0.01)
    return copy(env)
end


# 重みが 0 か １ に近づきすぎるとペナルティを与える。
function paramPenalty(a)
    threshold = 10f0
    strength = 1f0
    p = 0f0
    for v in a
        if abs(v) > threshold
            p += strength * (abs(v) - threshold)
        end
    end
    return p
end

# スパースじゃないとペナルティを与える。
function weightPenalty(a)
    strength = 1f0
    strength = 0.1f0    # 弱くする
    #strength = 0f0      # なくす
    ret = 0f0
    for v in a
        ret += sigmoid(v)
    end
    return strength * ret
end


function likelihood(batch, p::CPT4v2, env; noPenalty=false)
    #test
    #return sigmoid(bnet.params[3].b_ciu[1][1][1])

    #println("batch=", batch)
    #println("loss")
    sum = 0f0
    for vals in batch
        jp = calcJointP(p, vals)
        sum += -log(jp)
    end
    # if ! noPenalty
    #     sum += paramPenalty(m.w_ciud)
    #             + paramPenalty(m.w_cidk)
    #             + paramPenalty(m.w_ujdk)
    # end
    return sum
end

## 勾配法で学習。尤度 L(θ) = P(D|θ) = Σ_Z P(D,Z|θ)
# 下の層の値だけを訓練データとして与える。上の層はすべて隠れ変数。

# 上位層の変数で周辺化した対数同時分布の符号を変えたものを返す。
# uppers は上の層の変数のすべての値の組み合わせを保持した配列。
function hloss(batch, uppers, p::CPT4v2, env; noPenalty=false)
    ret = 0f0
    for input in batch
        prob = 0f0
        for hidden in uppers
            prob += calcJointP(p, vcat(hidden, input))
        end
        ret += -log(prob)
    end
    if ! noPenalty
        ret += weightPenalty(p.b_ciud) + weightPenalty(p.b_ujdk)
    end
    return ret
end


# 完全データでの学習
function trainWithCompleteData(source::CPT4v2, env::Dict)
    data = [env[:completeDataFilter](makeCompleteData(source))]
    @show length(data[1])
    @show data[1]
    @show env
    p = deepcopy(source)
    setRandomParams(p, env[:randParamFunc])
    println("Target network CPT before learning:")
    printCPT(p)
    println("Target network params before learning:")
    printParams(p)

    loss = env[:loss]
    opt = env[:opt]
    evalcb = throttle(() -> (@show loss(data[1], p, env, noPenalty=true)), 5)

    losses = []
    psa = []
    ps = params([p.b_ciud, p.b_ujdk])
    @epochs env[:epochs] begin
        train!((batch)->loss(batch,p,env), ps, zip(data), opt, cb=evalcb)
        @show loss1 = loss(data[1], p, env, noPenalty=true)
        push!(losses, loss1)
        push!(psa, collect(Iterators.flatten(ps.order)))
    end

    if false
        display(
            plot(
                plot(transpose(hcat(psa...))),
                plot(losses),
                layout=(2,1), size=(1000,1000))
        )
    end

    println("----")
    println("Source network CPT:")
    printCPT(source)
    println("Target network CPT:")
    printCPT(p)
    println("Source network params:")
    printParams(source)
    println("Target network params:")
    printParams(p)
end


# 下の層のみに値を与える学習
#完全データの場合とできるだけコード共有したかったがそれはあとまわし。
#動かすことを最優先。
function trainWithHiddenLayer(source::CPT4v2, env::Dict)
    @show data = [makeInputData(source)]
    @show length(data[1])
    @show data[1]
    @show env
    p = deepcopy(source)
    @show uppers = [pval for pval in valueCombination(p.uNodes, p.uUnits)]
    setRandomParams(p, env[:randParamFunc])
    println("Target network CPT before learning:")
    printCPT(p)
    println("Target network params before learning:")
    printParams(p)

    hloss = env[:hloss]  # loss for trainWithHiddenLayer
    opt = env[:opt]
    evalcb = throttle(() -> (@show hloss(data[1], uppers, p, env, noPenalty=true)), 5)

    losses = []
    psa = []
    ps = params([p.b_ciud, p.b_ujdk])
    @epochs env[:epochs] begin
        train!((batch)->hloss(batch,uppers,p,env), ps, zip(data), opt, cb=evalcb)
        @show loss1 = hloss(data[1], uppers, p, env, noPenalty=true)
        push!(losses, loss1)
        push!(psa, collect(Iterators.flatten(ps.order)))
    end

    if false
        display(
            plot(
                plot(transpose(hcat(psa...))),
                plot(losses),
                layout=(2,1), size=(1000,1000))
        )
    end

    println("----")
    println("Source network CPT:")
    printCPT(source)
    println("Target network CPT:")
    printCPT(p)
    println("Source network params:")
    printParams(source)
    println("Target network params:")
    printParams(p)
end


##

function learningTest0()
    env = defaultEnv()
    pat = makePatternNet()
    trainWithCompleteData(pat, env)
end

function learningTest1()
    env = defaultEnv()
    pat = defPattern(:[
        [X, 1]=>1
        [X, X]=>1
    ], vNodes=1,vUnits=2)
    trainWithCompleteData(pat, env)
end


function learningTest2()
    env = defaultEnv()
    pat = defPattern(:[
        [X, 1, 2]=>1
        [X, X, 2]=>1
        [1, X, Y]=>1
    ], vNodes=2,vUnits=3)
    trainWithCompleteData(pat, env)
end

function learningTest3()
    env = defaultEnv()
    #env[:epochs]=10000
    pat = defPattern(:[
        [X, 1]=>1
        [X, X]=>1
        [3, X]=>1
    ], vNodes=1,vUnits=3)
    trainWithHiddenLayer(pat, env)
end

function learningTest4()
    env = defaultEnv()
    #env[:epochs]=200
    pat = defPattern(:[
        [1, X, X]=>1
        [X, 2, X]=>1
        [X, X, 3]=>1
    ], vNodes=1,vUnits=3)
    trainWithCompleteData(pat, env)
end

function learningTest4skip()
    env = defaultEnv()
    env[:completeDataFilter] = skipSomeDataFilter
    #env[:epochs]=1000
    pat = defPattern(:[
        [1, X, X]=>1
        [X, 2, X]=>1
        [X, X, 3]=>1
    ], vNodes=1,vUnits=3)
    trainWithCompleteData(pat, env)
end

# vNodes を増やした場合のテスト用
function learningTest4n(n)
    env = defaultEnv()
    #env[:epochs]=200
    pat = defPattern(:[
        [1, X, X]=>1
        [X, 2, X]=>1
        [X, X, 3]=>1
    ], vNodes=n,vUnits=3)
    trainWithCompleteData(pat, env)
end

# なぜか NaN が出る。
function learningTest5()
    env = defaultEnv()
    #env[:completeDataFilter] = skipSomeDataFilter
    pat = defPattern(:[
        [1, X, X]=>1
        [X, 2, X]=>1
        [X, X, 3]=>1
        [3, 2, 1]=>1
    ], vNodes=1,vUnits=4)
    trainWithCompleteData(pat, env)
end

learningTest4()
#learningTest4skip()
