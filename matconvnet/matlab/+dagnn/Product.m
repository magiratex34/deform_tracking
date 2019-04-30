classdef Product < dagnn.ElementWise
  
  properties (Transient)
    numInputs
    prodMat
  end

  methods
    function outputs = forward(obj, inputs, params)
        
     outputs{1} = bsxfun(@times, inputs{1}, inputs{2});  
%       obj.numInputs = numel(inputs) ;
%       outputs{1} = inputs{1} ;
%       for k = 2:obj.numInputs
%         outputs{1} = outputs{1} .* inputs{k} ;
%       end
%       obj.prodMat = outputs{1};
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      
      [~, derInputs{1}, derInputs{2}] = bsxfun_der(@times, inputs{1}, inputs{2}, derOutputs{1});
        
%       for k = 1:obj.numInputs
%         derInputs{k} = (obj.prodMat ./ inputs{k}) .* derOutputs{1};
%       end
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Product layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = Product(varargin)
      obj.load(varargin) ;
    end
  end
end
