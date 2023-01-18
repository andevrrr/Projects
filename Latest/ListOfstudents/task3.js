const SOURCE_DATA = `http://localhost:3010/students`;

const XHR_DONE = 4;
const XHR_SUCCESS = 200;

function clickBut() {

  var ourRequest = new XMLHttpRequest();
  ourRequest.open('GET', SOURCE_DATA);
  onRequest.onload = function() {

    if (this.readyState === XHR_DONE && this.status === XHR_SUCCESS) {

      var list = document.getElementById('list_of_students');
      var data = JSON.parse(this.responseText);
      list.innerHTML = renderHTML(data);

    }else {
      console.log('ERROR' + ourRequest.status);
    }

  };

  ourRequest.send();

}

function renderHTML(data) {

  let html = '<table> \
              <tr> \
                  <th>Name</th>\
                  <th>Info</th>\
                  <th>Points</th>\
              </tr>';
  const length = Object.keys(data).length;

  for (let i = 0; i < length; i++) {
      let student = data[i];


      html += `<tr>
                      <td>${data[i].name}</td>
                      <td>${data[i].info}</td>
                      <td class="Point">${data[i].exercise_points}</td>
              </tr>`;
  }

  html += '</table>';
  return html;

}

function addStudent(data) {
    $.ajax({
        type: 'POST',
        url: SOURCE_DATA,
        dataType: 'JSON',
        data: data,
        success: function() {clickBut()},
        error: function(error) {
        console.log('Error! ' + error);
        }
    });
}

function handleStudent() {
    let name = document.getElementById('name').value;
    let info = document.getElementById('info').value;
    let points = document.getElementById('points').value;

    if (name == undefined || name == null || name == '') {
        name = 'Name';
    }
    if (info == undefined || info == null || info == '') {
        info = 'info';
    }
    if (points == undefined || points == null || points == '') {
        points = 'null';
    }

    let newStudent = {

        name: name,
        info: info,
        exercise_points: points
    }

    addStudent(newStudent);
}
