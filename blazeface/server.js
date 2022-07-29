var express = require('express');
var app = express();
var fs = require("fs");
var bodyParser = require('body-parser');
require("@tensorflow/tfjs-backend-cpu")
const returnTensors = false;

var decode = require('image-decode')

const blazeface = require('@tensorflow-models/blazeface');
const model = blazeface.load({modelUrl:"http://127.0.0.1:8080/model.json"});

var jsonParser = bodyParser.json();
 
app.get('/api/health-check', function (req, res) {
    res.send({"status": "ok"})
})

app.post('/api/face', jsonParser, function(req,res){
	console.log(req.body);
    let {data, width, height} = decode(Buffer.from(req.body.data, 'base64'))

    var image = {
        data: data,
        width: req.body.width,
        height: req.body.height
    }

    var predictions = model.then(function(res2){return res2.estimateFaces(image, returnTensors);})
    predictions.then(function(result){
        console.log({"status": "ok", "result": result})
        res.send({"status": "ok", "result": result});
    })
})

var server = app.listen(8081, function () {
  var host = server.address().address
  var port = server.address().port
  console.log("访问地址为 http://%s:%s", host, port)
})
