
const DATA_SOURCE = 'http://127.0.0.1:3010/students';

const XHR_DONE = 4;
const XHR_SUCCESS = 200;

function dataToHtmlRepresentation(dataObjects){
    let html = '<table>';
    const l = dataObjects.lenght;
    for (let i = 0; i < l; i++){
        const dataObject = dataObjects[i];
        html +=
        `
            <tr>
              <td>${dataObject.name}</td>
              <td>${dataObject.exercise_points}</td>
            </tr>

        `
    }
    html += '</table>';
    return html;
}

function getData0(){
    const xhr = new XMLHttpRequest();
    xhr.open('GET', DATA_SOURCE);

    xhr.onreadystatechange = function() {
        if(this.readyState === XHR_DONE){
            if (this.status === XHR_SUCCESS){
                const e = document.getElementById('data0');
                const data = JSON.parse(this.responseText);
                e.innerHTML = dataToHtmlRepresentation(data);
            }
            else {
                console.log("ERROR!!" + xhr.status);
            }
        }
    };

    xhr.send();
}

function getData1(){
    $.ajax({
        type: 'GET',
        url: DATA_SOURCE,
        dataType: 'JSON',
        success: function(data) {
            $('#data1').html(dataToHtmlRepresentation(data));
        },
        error: function(error){
            console.log('Error!' + error);
        }
    });
}

function getData2(){
    fetch(DATA_SOURCE).then(response => {return response.json()}).then(data => {
        document.getElementById('data2').innerHTML = dataToHtmlRepresentation(data);
    }).catch(error => console.error('AAAARRRRRRRGGGGG!' + error))
}

function getData3(){
    axios.get(DATA_SOURCE).then(response => {

        document.getElementById('data2').innerHTML = dataToHtmlRepresentation(response.data);
    }), error => console.error('ERROR' + error);
}

function postData(data){
    const response = fetch(DATA_SOURCE, {
        method: 'POST',
        headers: {
            'Content-Type' : 'application/json'
        },
        body: JSON.stringify(data)
    }).then(resp => {return resp.json();}).then(data => console.log(data));
}

function postDataButtonHandler(){
    let newStudentName = 'Johnson';
    let info = {
        name: 'Mira',
        exercise_points: 543
    };
    postData(info);
}
