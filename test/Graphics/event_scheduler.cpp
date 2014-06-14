#include "event_scheduler.hpp"
#include <QApplication>
#include <QDebug>

EventScheduler::EventScheduler()
  : receiver_(0)
  , event_(0)
{
  timer_.setSingleShot(true);
}

void EventScheduler::set_receiver(QObject *receiver)
{
  receiver_ = receiver;
}

void EventScheduler::schedule_event(QEvent *event, int delay_ms)
{
  event_ = event;
  timer_.start(delay_ms);
  connect(&timer_, SIGNAL(timeout()), this, SLOT(notify()));
}

void EventScheduler::notify()
{
  qApp->notify(receiver_, event_);
}