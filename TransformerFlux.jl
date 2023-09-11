using Flux,Flux.Functors,Statistics,LinearAlgebra,CUDA,cuDNN


struct PositionalEncoding{T,U}
    d_model ::Int
    seq_len ::Int
    dropout ::T
    pe ::U
end
function PositionalEncoding(d_model::Int,seq_len::Int,dropout::T) where T<:Real
    pe = zeros(Float32,d_model,seq_len)
    position = reshape(1:seq_len,(1,seq_len))
    div_term = exp.(collect(1:2:d_model) .* (-log(10_000)/d_model))
    pe[1:2:end,:] = sin.(position .* div_term)
    pe[2:2:end,:] = cos.(position .* div_term)
    pe = reshape(pe,(d_model,seq_len,1))
    d = Flux.Dropout(dropout)
    PositionalEncoding{typeof(d),typeof(pe)}(
        d_model,
        seq_len,
        d,
        pe
    )
end
function (posenc ::PositionalEncoding)(x)
    x = x .+ posenc.pe
     posenc.dropout(x)
end
@functor PositionalEncoding (d_model,seq_len,dropout) 


struct InputEmbeding
    emb ::Embedding
    pos ::PositionalEncoding
end
function InputEmbeding(vocab_size,d_model,seq_len,dropout)
    InputEmbeding(
        Flux.Embedding(vocab_size=>d_model),
        PositionalEncoding(d_model,seq_len,dropout)
    )
end
function (T::InputEmbeding)(x)
    y = T.emb(x)
    T.pos(y) .+ y
end
@functor InputEmbeding


struct EncoderLayer
    MHA::Flux.MultiHeadAttention
    NL1 ::Flux.LayerNorm
    FF ::Flux.Chain
    NL2 ::Flux.LayerNorm
end
function EncoderLayer(d_model,d_ff,h,drop1=0.1,drop2=0.1)
    EncoderLayer(
        Flux.MultiHeadAttention(d_model,nheads=h,dropout_prob=drop1),
        Flux.LayerNorm(d_model),
        Flux.Chain(
            Dense(d_model=>d_ff,relu),
            Flux.Dropout(drop2),
            Dense(d_ff=>d_model)
        ),
        Flux.LayerNorm(d_model),
    )
end
function (enc::EncoderLayer)(x)
    y = copy(x)
    yc = copy(y)
    y,att = enc.MHA(y)
    y = enc.NL1(y .+ yc)
    yc = copy(y)
    y = enc.FF(y)
    y = enc.NL2(y .+ yc)
end
@functor EncoderLayer

struct Encoder
    encLayers::Vector{EncoderLayer}
end
function Encoder(nlayer,d_model,d_ff,h;drop1=0.1,drop2=0.1)
    Encoder([
        EncoderLayer(d_model,d_ff,h,drop1,drop2) for _ in 1:nlayer
    ])
end
function (E::Encoder)(x)
    y = E.encLayers[1](x)
    for EL in E.encLayers[2:end]
        y = EL(y)
    end
    y
end
@functor Encoder

struct DecoderLayer
    MHA1 ::Flux.MultiHeadAttention
    NL1 ::Flux.LayerNorm
    MHA2 ::Flux.MultiHeadAttention
    NL2 ::Flux.LayerNorm
    FF ::Chain
    NL3 ::Flux.LayerNorm
end
function DecoderLayer(d_model,d_ff,h1,h2,drop1=0.1,drop2=0.1)
    DecoderLayer(
        Flux.MultiHeadAttention(d_model,nheads = h1,dropout_prob = drop1),
        Flux.LayerNorm(d_model),
        Flux.MultiHeadAttention(d_model,nheads=h2,dropout_prob = drop2),
        Flux.LayerNorm(d_model),
        Chain(
            Dense(d_model=>d_ff,relu),
            Dense(d_ff=>d_model)
        ),
        Flux.LayerNorm(d_model)
    )
end
function (D::DecoderLayer)(x,y;device=cpu)
    z = copy(y)
    zc = copy(z)
    z,as1 = D.MHA1(z,mask=device(make_causal_mask(z, dims=2)))
    z = D.NL1(z .+ zc)
    zc = copy(z)
    z,as2 = D.MHA2(x,x,z)
    z = D.NL2(z .+ zc)
    zc = copy(z)
    z = D.FF(z)
    z = D.NL3(z .+ zc)
end
@functor DecoderLayer

struct Decoder
    decLayers::Vector{DecoderLayer}
end
function Decoder(nlayer,d_model,d_ff,h1,h2;drop1=0.1,drop2=0.1)
    Decoder([
        DecoderLayer(d_model,d_ff,h1,h2,drop1,drop2) for _ in 1:nlayer
    ])
end
function (D::Decoder)(x,y;device=cpu)
    z = copy(y)
    for DL in D.decLayers
        z = DL(x,z,device = device)
    end
    z
end
@functor Decoder

struct Transformer
    enc ::Encoder
    dec ::Decoder
end
function Transformer(d_model,dropout,nlayer,d_ff,h)
    Transformer(
        Encoder(nlayer,d_model,d_ff,h;drop1 = dropout,drop2 = dropout),
        Decoder(nlayer,d_model,d_ff,h,h;drop1 = dropout,drop2 = dropout)
    )
end
function (T::Transformer)(x;device = cpu)
    y = T.enc(x)
    y = T.dec(y,device(ones(Float32,size(y)...)),device=device)
end
@functor Transformer

struct TransformerTokenizer
    tok ::InputEmbeding
    T ::Transformer
    final ::Chain
end
function TransformerTokenizer(vocab_size,d_model,seq_len,dropout,nlayer,d_ff,h)
    TransformerTokenizer(
        InputEmbeding(vocab_size,d_model,seq_len,dropout),
        Transformer(d_model,dropout,nlayer,d_ff,h),
        Chain( Dense(d_model=>vocab_size), softmax )
    )
end
function (TT::TransformerTokenizer)(x)
    y = TT.tok(x) 
    y = TT.T(y)
    TT.final(y)
end
@functor TransformerTokenizer


seq = "hey guys, how are you ?"
vocabl = 'a':'z' |> collect
pushfirst!(vocabl,' ')
for i in seq
    if !(i in vocabl)
        push!(vocabl,i)
    end
end
vocab = Dict(vocabl[i]=>i for i in eachindex(vocabl))
invvocab = Dict(i=>vocabl[i] for i in eachindex(vocabl))
x = Flux.onehotbatch([vocab[i] for i in seq],1:length(vocab))
x = reshape(x,(size(x)...,1))

vocab_size = length(vocab)
d_model = 16
seq_len = length(seq)
h = 8
dropout = 0.1
nlayer = 4
d_ff = 16
tok = InputEmbeding(vocab_size,d_model,seq_len,dropout)
model = TransformerTokenizer(vocab_size,d_model,seq_len,dropout,nlayer,d_ff,h)


function getOutput(model,x)
	String(
		[
			invvocab[
				( 
					argmax.(
						eachcol(
							model(x)[:,:,1]
						)
					) 
				)[i] 
			] for i in 1:size(x,2)
		]
	)
end

getOutput(model,x)

Flux.params(model) .|> length |> sum

"""

This is just to show how to train a transformer, of course, you should train a full TransformerTokenizer, meaning build a "real" vocab and a "real" batch of sentences adn use TransformerTokenizer of course.

"""

device = cpu # enouth in this case, but one can try gpu if wanted.
model_transformer = Transformer(d_model,dropout,nlayer,d_ff,h) |> device

x_train = rand(Float32,d_model,seq_len,100) |> device
y_train = rand(Float32,d_model,seq_len,100) |>device
datas = Flux.DataLoader((x_train,y_train),batchsize = 10,shuffle=true)
loss(model,x,y;device=device) = Flux.mse(model(x,device=device),y)
opt = Flux.setup(Adam(0.01),model_transformer)


@show loss(model_transformer,x_train,y_train,device=device)
for _ in 1:10
    Flux.train!((m,x,y)->loss(m,x,y,device=device),model_transformer,[(x_train,y_train)],opt)
    @show loss(model_transformer,x_train,y_train,device=device)
end
@show loss(model_transformer,x_train,y_train,device=device)
