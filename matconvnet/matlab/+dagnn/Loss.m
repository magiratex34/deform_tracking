classdef Loss < dagnn.ElementWise
  properties
    loss = 'softmaxlog'
    ignoreAverage = false
    K_loss = false
    opts = {}
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      if numel(inputs) == 3
          % cut the k-th part out of x
          k = inputs{3};
          outputs{1} = vl_nnloss(inputs{1}(:,:,2*k-1:2*k,:), inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
      else
          outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
      end;
      
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if numel(inputs) == 3
          % cut the k-th part out of x
          k = inputs{3};
          derInputs{1} = gpuArray(zeros(size(inputs{1}),'single'));
          derInputs{1}(:,:,2*k-1:2*k,:) = vl_nnloss(inputs{1}(:,:,2*k-1:2*k,:), inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;
          derInputs{2} = [];
          derInputs{3} = [];
          derParams = {} ;
      else
          derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;
          derInputs{2} = [] ;
          derParams = {} ;
      end;
      
%       if obj.K_loss
%           k = inputs{3};
%           
%           derInputs{1}(:,:,2*k-1:2*k,:) = derloss;
%           derInputs{3} = [];
%       else
%           derInputs{1} = derloss;
%       end;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end
    
    function accumulateAverage(obj, inputs, outputs)
      if obj.ignoreAverage, return; end;
      n = obj.numAveraged ;
      m = n + size(inputs{1}, 1) *  size(inputs{1}, 2) * size(inputs{1}, 4);
      obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
