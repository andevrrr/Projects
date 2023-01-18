
// changing the pages

// increasing the font-size when clicking the button

function page(page){
    var style = `
    .menu_btn:nth-child(`+ page +`){
        animation: menu_btn 1s ease-in forwards;
      }
      
      @keyframes menu_btn {
      
        50%{
          font-size: 14px;
        }
        75%{
            font-size: 15px;
          }
        100% {
            margin-top: 15px;
          font-size: 16px;
        }
      }

    `
    for (var i = 1; i < 4; i++){
        if (i != page){
            style += `.menu_btn:nth-child(`+ i +`){
                animation: menu_btn2 1s ease-in forwards;
              }
              
              @keyframes menu_btn2 {
              
                100% {
                  font-size: 13px;
                }
              }`
        }
    }
        var styleSheet = document.createElement("style")
        styleSheet.innerText = style;
        document.head.appendChild(styleSheet)
}

function show1(shown, hidden1, hidden3) {
    document.getElementById(shown).style.display='block';
    document.getElementById(hidden1).style.display='none';
    document.getElementById(hidden3).style.display='none';
    page(1)
    return false;
}

function show2(shown, hidden1,  hidden3) {
    document.getElementById(shown).style.display='block';
    document.getElementById(hidden1).style.display='none';
    document.getElementById(hidden3).style.display='none';
    page(2)
    return false;
}
function show3(shown, hidden1,  hidden3) {
    document.getElementById(shown).style.display='block';
    document.getElementById(hidden1).style.display='none';
    document.getElementById(hidden3).style.display='none';
    page(3)
    return false;
}



////


let url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/'
var count = 30;
var content = "";

var current = new Date();
//

// these are for the chart
var labels = [0,1,2]
var data_data = [1,2,3,4]

// chart
/*
var count_minutes = 0;
var sum = 0;
var average = 0;
*/
var ctx = document.getElementById('myChart').getContext("2d");

var data = {
    labels, 
    datasets: [{
        data: data_data,
        label: "Weather data",
    }]
}

var config = {
    type: 'line',
    data:  data,
    options: {
        responsive: true,
        /*scales: {
            y: {
                ticks: {
                    callback: function (value){
                        return  value + " min";
                    }
                }
            }
        } */
    },
};
var chart = new Chart(ctx, config);


//

// Here i display the table and get data from the json link

function fetch_data(){
    $.getJSON(url, function(json) {
        labels.length = 0;
        let text = "";
        let text2 ="";
        let day = "<th>Day</th>";
        let num = "<th>Device</th>";
        if(count > 30 || count == 24){
            num = "";
        }
        text += "<thead><tr><th>Number</th>" + num + "<th>Date</th><th>Time</th><th>Type</th><th>Value</th></tr></thead>";

        for (i = 0; i < count; i++) {
            
                if (count == 30){
                    document.getElementById('dropdown').style.display='none';
                    document.getElementById('chart').style.display='none';
                    if (i == count/2){
                        text2 = text;
                        text = "";
                    }
                    text += "<tr><td>" + (i+1) + "</td>";
                    var name = Object.keys(json[i].data);
                    text += "<td>" + json[i].id + "</td>";
                    text += "<td>" + new Date(json[i].date_time).getDate() + "." + new Date(json[i].date_time).getMonth() + "."+ new Date(json[i].date_time).getFullYear() + "</td>";
                    text += "<td>" + new Date(json[i].date_time).getHours() + ":" + new Date(json[i].date_time).getMinutes() + ":" + new Date(json[i].date_time).getSeconds() + "</td>";
                    text += "<td>" + name + "</td>";
                    text += "<td>" + json[i].data[name] + "</td></tr>";
                    
                } else{
                    document.getElementById('dropdown').style.display='block';
                    document.getElementById('chart').style.display='block';
                    if (i == count/2){
                        text2 = text;
                        text = "";
                    }
                    text += "<tr><td>" + (i+1) + "</td>";

                    if(count>30 || count == 24){
                        text += "<td>" + new Date(json[i].date_time).getDate() + "." + new Date(json[i].date_time).getMonth() + "."+ new Date(json[i].date_time).getFullYear()  + "</td>";
                    }else{
                        text += "<td>" + json[i].device_id + "</td>";
                        text += "<td>" + new Date(json[i].date_time).getDate() + "." + new Date(json[i].date_time).getMonth() + "."+ new Date(json[i].date_time).getFullYear()  + "</td>";
                    }
                    text += "<td>" + new Date(json[i].date_time).getHours() + ":" + new Date(json[i].date_time).getMinutes() + ":" + new Date(json[i].date_time).getSeconds() + "</td>";
                    text += "<td>" + content + "</td>";
                    text += "<td>" + json[i][content] + "</td></tr>";
                    if(count>30){
                        labels[i] = new Date(json[i].date_time).getDate();
                    }else if (count == 24){
                        labels[i] = new Date(json[i].date_time).getHours() + "h";
                    }else{
                        labels[i] = new Date(json[i].date_time).getSeconds();
                    }
                    data_data[i] = json[i][content];
                    if (content == "temperature" || content == "wind_speed"){
                        config.type = 'bar'

                    }else {
                        config.type = 'line'
                    }
                }
            
        }
        /* this is counting the average 
        if (current.getMinutes != 00){
            data_data[count_minutes] = json[i][content];
            count_minutes++;
            sum = 0;
            average = 0;
        } else {
            for (var i = 0; i < count_minutes; i++){
                sum += data_data[i]; 
            }
            average = sum/count_minutes;
            count_minutes = 0;
        }*/


        

        text += "</tbody>"
        document.getElementById("showData").innerHTML = text2;
        document.getElementById("showData2").innerHTML = text;
        chart.update();
        
        
    });
}

if (count == 24 || count > 30 ){
    fetch_data();
}else{
    setInterval(function() { // updates the table every second
        fetch_data();                     
    }, 1000);
}  

// Below the code is about clicking on buttons to filter the table

var btnContainer = document.getElementById("myBtnContainer");
var btns = document.getElementsByClassName("btn");
btns[0].addEventListener("click", function() {
    url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/'
    count = 30;

 });
btns[1].addEventListener("click", function() {
    url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/wind_speed'
    count = 20;
    content = "wind_speed"

 });
 btns[2].addEventListener("click", function() {
    url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/temperature'
    count = 20;
    content = "temperature"
 });
 btns[3].addEventListener("click", function() {
     url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/rain'
    count = 20;
    content = "rain"

 });
 btns[4].addEventListener("click", function() {
    url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/light'
    count = 20;
    content = "light"

 });
 btns[5].addEventListener("click", function() {
    url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/humidity_out'
    count = 20;
    content = "humidity_out"

 });
 btns[6].addEventListener("click", function() {
    url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/humidity_in'
    count = 20;
    content = "humidity_in"
 });

///// Droped down menu
var t = "";

 var dropdown_content = document.getElementsByClassName("dropdown_content");
 var menus = document.getElementsByClassName("menu");
 menus[0].addEventListener("click", function() {
    t = "";
    count = 20;
    filter_click(t);
});
 menus[1].addEventListener("click", function() {

     t = 23;
     count = 24;
     filter_click(t);
 });
 menus[2].addEventListener("click", function() {
    t = 47;
    count = 48;
    filter_click(t);
});
menus[3].addEventListener("click", function() {
    t = 71;
    count = 72;
    filter_click(t);
});
menus[4].addEventListener("click", function() {
    t = 167;
    count = 168;
    filter_click(t);
});


 function filter_click(x){
    url = 'http://webapi19sa-1.course.tamk.cloud/v1/weather/' + content + "/" + x
 }


 /// forecast widget
 $.getJSON('https://webapi19sa-1.course.tamk.cloud/v1/weather/temperature/24', function(json) {
    document.getElementById("degree").innerHTML = Math.round(json[24].temperature) + "&#176;";
 })
 $.getJSON('https://webapi19sa-1.course.tamk.cloud/v1/weather/humidity_in/24', function(json) {
    document.getElementById("humidity").innerHTML = Math.round(json[24].humidity_in) + "&#37;";
 })
 $.getJSON('https://webapi19sa-1.course.tamk.cloud/v1/weather/wind_speed/24', function(json) {
    document.getElementById("wind").innerHTML = Math.round(json[24].wind_speed) + "m/s";
 })
 $.getJSON('https://webapi19sa-1.course.tamk.cloud/v1/weather/light/24', function(json) {
    document.getElementById("light").innerHTML = Math.round(json[24].light) + "&#37;";
 })

 // change the background-image depanding on the time

if (current.getHours() >= 0 && current.getHours() <= 5){
    document.getElementById('forecast').style.background = "linear-gradient(to top,  rgb(0, 86, 167), rgb(0, 51, 145),  rgb(0, 0, 78))";
    document.getElementById('sun').src = "https://cdn-icons-png.flaticon.com/512/740/740878.png";
}else if (current.getHours() >= 21 && current.getHours() <= 23){
    document.getElementById('forecast').style.background = "linear-gradient(to top,  rgb(51, 151, 245), rgb(21, 92, 223),  rgb(2, 2, 119))";
    document.getElementById('sun').src = "https://cdn-icons-png.flaticon.com/512/740/740878.png";
}else if (current.getHours() >= 6 && current.getHours() <= 20){
    document.getElementById('forecast').style.background = "linear-gradient(to top,  rgb(85, 178, 255),  rgb(144, 220, 255), rgb(255, 255, 104))";
    document.getElementById('sun').src = "https://cdn-icons-png.flaticon.com/512/869/869869.png";
} 



setInterval(function() { 
    var hour = new Date().getHours();
    var min = new Date().getMinutes();
    if (hour < 10 ){
        hour = "0" + hour
    }
    if (min < 10 ){
        min = "0" + min
    }
    var time = hour + ":" + min; 
    document.getElementById('forecast_time').innerHTML =  time            
}, 1000);
