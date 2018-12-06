classdef philipNN
    %PHILIPNN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        y0;
        y1;
        x0;
        x1;
    end
    
    methods
        function obj = philipNN(x0,x1,y0,y1)
            %PHILIPNN Construct an instance of this class
            %   Detailed explanation goes here
            obj.x0 = x0;
            obj.x1 = x1;
            obj.y0 = y0;
            obj.y1 = y1;
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end

