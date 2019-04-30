classdef DeformGridGenerator < dagnn.Layer

 properties
     Ho = 0;
     Wo = 0;
 end

  properties (Transient)
    xxyy ;
  end

  methods
    function outputs = forward(obj, inputs, ~)
    
      useGPU = isa(inputs{1}, 'gpuArray');
      A = inputs{1};
      nbatch = size(A,4);
      A = reshape(A, 2, obj.Ho*obj.Wo, nbatch);
      A = permute(A,[2,1,3]);
      if isempty(obj.xxyy)
        obj.initGrid(useGPU);
      end
      g = bsxfun(@plus, obj.xxyy, A);
      g = reshape(g, obj.Wo, obj.Ho, 2, nbatch);
      g = permute(g, [3,2,1,4]);

      outputs = {g};
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)

      useGPU = isa(derOutputs{1}, 'gpuArray');
      dY = derOutputs{1};
      nbatch = size(dY,4);

      % cudnn compatibility:
      dY = permute(dY, [3,2,1,4]);
      
      dA = reshape(dY, obj.Ho*obj.Wo, 2, nbatch);
      if useGPU, dA = gpuArray(dA); end

      dA = permute(dA, [2,1,3]);
      dA = reshape(dA, size(inputs{1}));
      derInputs = {dA};
      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      nBatch = inputSizes{1}(4);
      outputSizes = {[2, obj.Ho, obj.Wo, nBatch]};
    end

    function obj = DeformGridGenerator(varargin)
      obj.load(varargin) ;
      % get the output sizes:
      obj.Ho = obj.Ho ;
      obj.Wo = obj.Wo ;
      obj.xxyy = [] ;
    end

    function obj = reset(obj)
      reset@dagnn.Layer(obj) ;
      obj.xxyy = [] ;
    end

    function initGrid(obj, useGPU)
      % initialize the grid:
      % this is a constant
      xi = linspace(-1, 1, obj.Ho);
      yi = linspace(-1, 1, obj.Wo);

      [yy,xx] = meshgrid(xi,yi);
      xxyy = [yy(:), xx(:)] ; % Mx2
      if useGPU
        xxyy = gpuArray(xxyy);
      end
      obj.xxyy = xxyy ;
    end

  end
end
