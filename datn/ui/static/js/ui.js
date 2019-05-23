var $messages = $('.messages-content');
var date, hour, minute;
var image_temp = '<img src="/static/images/avatar.jpeg" />';
var avatar_temp = '<figure class="avatar">' + image_temp + '</figure>';

var Welcome = [
  'Chào bạn, shop có thể giúp gì được bạn ạ',
  'Shop thời trang HniLa xin chào bạn. :)',
  'Bạn cần shop tư vấn gì không ạ?',
  ':)'
]

var Unknown = [
  'Mình chưa hiểu ý bạn, bạn nói lại được không ạ',
  'Điều này mình chưa biết, để mình hỏi lại quản lý. Có gì mình sẽ liên hệ lại cho bạn sớm ạ.',
  'Bạn chờ một lát nhé. Bên mình sẽ có người trả lời bạn sau ạ.'
]

function updateScrollbar() {
  /* Update scroll bar to bottom of conversation */
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {});
}

function showDate() {
  /* Show the difference time of the last message */
  date = new Date();
  if (minute != date.getMinutes()) {
    minute = date.getMinutes();
    hour = date.getHours();
    time = '<div class="timestamp">' + date.getHours() + ':' + minute + '</div>';
    $(time).appendTo($('.message:last'));
  }
}

function welcomeMessage() {
  /* Random message welcome in first time */
  var random = Math.floor(Math.random() * Welcome.length);
  var random_msg = Welcome[random];

  renderReply(random_msg);
}

function unknownMessage() {
  /* Random message unknown when requestReply return error  */
  var random = Math.floor(Math.random() * Unknown.length);
  var random_msg = Unknown[random];

  renderReply(random_msg);
}

function sendMessage() {
  var message = $('.message-input').val();
  if (!$.trim(message)) {
    return false;
  }
  var msg_temp = '<div class="message message-personal">' + message + '</div>';
  $(msg_temp).appendTo($('.mCSB_container')).addClass('new');
  $('.message-input').val(null);

  showDate();
  updateScrollbar();
  getReply(message);
}

function getReply(message) {
  if ($('.message-input').val() != '') {
    return false;
  }

  // Render loading
  var loading_temp = '<div class="message loading new">' + avatar_temp + '<span></span></div>';
  $(loading_temp).appendTo($('.mCSB_container'));

  requestReply(message);
}

function requestReply(message) {
  /** Request reply message from chatbot
   * @param: {String} message: question/message from user
   * @return: NULL
  **/

  $.ajax({
    url: "/get_reply",
    method: "GET",
    data: {
      message: message
    },
    success: function(response) {
      renderReply(response);
    },
    error: function(xhr) {
      unknownMessage();
    }
  })
}

function renderReply(response) {
  /** Render reply message from chatbot to user
   * @param: {String} response: answer/reply message want to send to user
   * @return: NULL
  **/

  setTimeout(function() {
    $('.message.loading').remove();
    var message_temp = '<div class="message new">' + avatar_temp + response + '</div>';
    $(message_temp).appendTo($('.mCSB_container')).addClass('new');
    showDate();
    updateScrollbar();
  }, 1000 + (Math.random() * 20) * 100);
}


$(window).on('load', function() {
  $messages.mCustomScrollbar();
  setTimeout(function() {
    welcomeMessage();
  }, 100);
});

$('#sendMessage').click(function() {
  sendMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    sendMessage();
  }
});
