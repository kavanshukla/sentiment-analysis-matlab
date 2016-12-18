function [pool_res,cache] = CNN(X,filter_len,wConv, bConv,cache,pool_res)
    for k = 1:filter_len
            if(k < length(X(:,1)))
                conv = vl_nnconv(X,wConv{k},bConv{k});

                relu = vl_nnrelu(conv);

                sizes = size(conv);

                pool = vl_nnpool(relu,[sizes(1),1]);
                cache{2,k} = relu;
                cache{1,k} = conv;
                pool_res{k} = pool;
            end
     end