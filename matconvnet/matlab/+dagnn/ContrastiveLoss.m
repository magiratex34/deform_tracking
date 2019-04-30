classdef ContrastiveLoss < dagnn.Loss
  properties
    margin = 1;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      switch numel(inputs)
        case 3
          outputs{1} = vl_nncontrloss(inputs{:}, 'margin', obj.margin);
        case 4
          outputs{1} = vl_nncontrloss(inputs{1:3}, 'margin', inputs{4});
        otherwise
          error('Invalid number of inputs.');
      end
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      switch numel(inputs)
        case 3
          [dzdx1, dzdx2] = vl_nncontrloss(inputs{:}, derOutputs{1}, 'margin', obj.margin);
        case 4
          [dzdx1, dzdx2] = vl_nncontrloss(inputs{1:3}, derOutputs{1}, 'margin', inputs{4});
        otherwise
          error('Invalid number of inputs.');
      end
      derInputs = {dzdx1, dzdx2, []};
      derParams = {} ;
    end
    
    function obj = ContrastiveLoss(varargin)
      obj.load(varargin{:}) ;
    end
  end
end

