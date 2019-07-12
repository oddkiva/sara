#include <QGuiApplication>

#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QAspectEngine>

#include <Qt3DRender/qrenderaspect.h>
#include <Qt3DRender/QCamera>
#include <Qt3DRender/QMaterial>

#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DExtras/QTorusMesh>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DExtras/QPhongMaterial>


Qt3DCore::QEntity* createTestScene()
{
    Qt3DCore::QEntity* root = new Qt3DCore::QEntity;
    Qt3DCore::QEntity* torus = new Qt3DCore::QEntity(root);

    Qt3DExtras::QTorusMesh* mesh = new Qt3DExtras::QTorusMesh;
    mesh->setRadius(5);
    mesh->setMinorRadius(1);
    mesh->setRings(100);
    mesh->setSlices(20);

    Qt3DCore::QTransform* transform = new Qt3DCore::QTransform;
//    transform->setScale3D(QVector3D(1.5, 1, 0.5));
    transform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1,0,0), 45.f ));

    Qt3DRender::QMaterial* material = new Qt3DExtras::QPhongMaterial(root);

    torus->addComponent(mesh);
    torus->addComponent(transform);
    torus->addComponent(material);

    return root;
}

int main(int argc, char* argv[])
{
    QGuiApplication app(argc, argv);
    Qt3DExtras::Qt3DWindow view;
    Qt3DCore::QEntity* scene = createTestScene();

    // camera
    Qt3DRender::QCamera *camera = view.camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(0, 0, 40.0f));
    camera->setViewCenter(QVector3D(0, 0, 0));

    // manipulator
    Qt3DExtras::QOrbitCameraController* manipulator = new Qt3DExtras::QOrbitCameraController(scene);
    manipulator->setLinearSpeed(50.f);
    manipulator->setLookSpeed(180.f);
    manipulator->setCamera(camera);

    view.setRootEntity(scene);
    view.show();

    return app.exec();
}
