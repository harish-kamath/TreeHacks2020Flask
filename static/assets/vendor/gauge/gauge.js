
$(function() {
    "use strict";
    // ==============================================================
    // Guage 1
    // ==============================================================
    var opts = {
        angle: 0, // The span of the gauge arc
        lineWidth: 0.32, // The line thickness
        radiusScale: 0.7, // Relative radius
        pointer: {
            length: 0.6, // // Relative to gauge radius
            strokeWidth: 0.088, // The thickness
            color: '#2e2f39' // Fill color
        },
        limitMax: false, // If false, max value increases automatically if value > maxValue
        limitMin: false, // If true, the min value of the gauge will be fixed
        colorStart: '#e4e4ee', // Colors
        colorStop: '#5969ff', // just experiment with them
        strokeColor: '#e4e4ee', // to see which ones work best for you
        generateGradient: true,
        highDpiSupport: true, // High resolution support

    };
    var target = document.getElementById('gauge1'); // your canvas element
    var gauge = new Gauge(target).setOptions(opts); // create sexy gauge!
    gauge.maxValue = 3000; // set max gauge value
    gauge.setMinValue(0); // Prefer setter over gauge.minValue = 0
    gauge.animationSpeed = 76; // set animation speed (32 is default value)
    gauge.set(1850); // set actual value


// ==============================================================
    // Guage 2
    // ==============================================================














    });
