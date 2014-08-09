#include <QObject>
#include <QTimer>

class EventScheduler: public QObject
{
  Q_OBJECT
private:
  QTimer timer_;
  QObject *receiver_;
  QEvent *event_;

public:
  EventScheduler();
  void set_receiver(QObject *receiver);

public slots:
  void schedule_event(QEvent *event, int delay_ms);
  void notify();
};