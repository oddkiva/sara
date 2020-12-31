//
//  Boost2XCTest.mm
//
//  Created by Oscar Hierro - https://github.com/oscahie
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>
#import <objc/runtime.h>

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_parameters.hpp>
#include <boost/test/tree/traverse.hpp>
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test_log_formatter.hpp>
#include <boost/test/execution_monitor.hpp>

using namespace boost::unit_test;


/**
 * A custom log formatter whose only purpose is to monitor for failed assertions and uncaught exceptions within the Boost tests
 * in order to report them to the corresponding XCTest test case, pinpointing the actual line of code that caused it when possible.
 */
struct xctest_failure_reporter : unit_test_log_formatter {

  void set_current_xctestcase(XCTestCase *tc) { _xctest = tc; }

  void log_start(std::ostream &, counter_t) {}
  void log_finish(std::ostream &) {}
  void log_build_info( std::ostream&, bool) {}


  void test_unit_start(std::ostream &, test_unit const &) {}
  void test_unit_finish(std::ostream &, test_unit const &, unsigned long) {}
  void test_unit_skipped(std::ostream &, test_unit const &, const_string) {}
  void test_unit_skipped(std::ostream &, test_unit const &) {}
  void test_unit_aborted(std::ostream &, test_unit const &) {}

  /* Called for uncaught exceptions within the test cases (or their fixtures) */
  void log_exception_start(std::ostream&,
                           log_checkpoint_data const&,
                           boost::execution_exception const& ex)
  {
    boost::execution_exception::location const& loc = ex.where();
    if (!loc.m_file_name.empty())
    {
      _sourceFilePath = [[NSString alloc] initWithBytes:loc.m_file_name.begin()
                                                 length:loc.m_file_name.size()
                                               encoding:NSUTF8StringEncoding];
      _sourceFileLine = loc.m_line_num;
    }
    _failureDescription << ex.what();
    _failureHasBeenReported = true;
  }

  void log_exception_finish(std::ostream &)
  {
    if (_failureHasBeenReported)
    {
      record_xctest_failure();
    }
  }

  /* Called for failed assertions within the test cases */
  void log_entry_start(std::ostream &,
                       log_entry_data const &logEntryData,
                       log_entry_types logEntryType)
  {
    switch (logEntryType)
    {
    case BOOST_UTL_ET_ERROR:
            case BOOST_UTL_ET_FATAL_ERROR:
      {
        _sourceFilePath = @(logEntryData.m_file_name.c_str());
        _sourceFileLine = logEntryData.m_line_num;
        _failureHasBeenReported = true;
        break;
      }
            default:
      break;
    }
  }

  void log_entry_value(std::ostream &, const_string value)
  {
    _failureDescription << value;
  }

  void log_entry_value(std::ostream &, lazy_ostream const &value)
  {
    _failureDescription << value;
  }

  void log_entry_finish(std::ostream &)
  {
    if (_failureHasBeenReported)
    {
      record_xctest_failure();
    }
  }

  void entry_context_start(std::ostream &, log_level) {}
  void log_entry_context(std::ostream &, log_level, const_string) {}

  void entry_context_finish(std::ostream &, log_level) {}

  void set_log_level(log_level) {};
  log_level get_log_level() const { return log_all_errors; }

private:

  /* Record the failure in the XCTestCase we're currently executing  */
  void record_xctest_failure()
  {
    NSString *description = _failureDescription.str().size() ? [[NSString stringWithUTF8String:_failureDescription.str().c_str()] stringByStandardizingPath] : NULL;

    assert(_xctest);
    [_xctest recordFailureWithDescription:description
                                   inFile:_sourceFilePath
                                   atLine:_sourceFileLine
                                 expected:YES];

    _failureHasBeenReported = false;
    _sourceFileLine = 0;
    _sourceFilePath = NULL;
    _failureDescription.str("");
  }

  XCTestCase *_xctest;
  bool _failureHasBeenReported;
  NSString *_sourceFilePath;
  NSUInteger _sourceFileLine;
  std::stringstream _failureDescription;
};

xctest_failure_reporter *xcreporter;


static bool boost_init_function()
{
  boost::unit_test::unit_test_log.set_formatter(xcreporter);
  return true;
}

/**
 * A test tree visitor that dynamically creates a new XCTestCase object for each Boost test case
 */
class xctest_dynamic_registerer : public test_tree_visitor
{
public:

  /* Dynamically create a new class for the current test suite */
  virtual bool test_suite_start(test_suite const& ts)
  {
    const char *testClassName = ts.p_name->c_str();

    if (NSClassFromString(@(testClassName)) == NULL)
    {
      testSuiteClass = objc_allocateClassPair([XCTestCase class], ts.p_name->c_str(), 0);
      assert(testSuiteClass);

      objc_registerClassPair(testSuiteClass);
    }

    return true;
  }

  /* Add a new test method to the current test suite */
  virtual void visit(const test_case& tu)
  {
    NSString *testCaseName = @(tu.p_name->c_str());
    NSString *testSuiteName = NSStringFromClass(testSuiteClass);

    IMP imp = imp_implementationWithBlock(^void (id self /* XCTestCase *xctest */) {
                                          xcreporter = new xctest_failure_reporter();
                                          xcreporter->set_current_xctestcase(self);

                                          // craft the arguments to instruct Boost to run _only_ the current test case
                                          NSString *runTestArgs = [NSString stringWithFormat:@"--run_test=%@/%@", testSuiteName, testCaseName];
                                          const char *argv[2] = { "dummy-arg", runTestArgs.UTF8String };
                                          int argc = 2;

                                          // execute the test case
                                          unit_test_main(&boost_init_function, argc, (char**)argv);
                                          });

    SEL selector = sel_registerName(testCaseName.UTF8String);
    BOOL added = class_addMethod(testSuiteClass, selector, imp, "v@:");
    if (!added)
    {
      NSLog(@"Failed to add test case method '%@', this method may already exist in the test suite '%@'", testCaseName, testSuiteName);
      assert(false);
    }
  }

private:
  Class testSuiteClass;
};


/**
 * Bootstraps the generation of the XCTest test cases
 *
 * This approach was taken from https://github.com/mattstevens/xcode-googletest/
 */
@interface BoostTestLoader : NSObject
@end

@implementation BoostTestLoader

+ (void)load
{
  NSBundle *bundle = [NSBundle bundleForClass:self];

  [[NSNotificationCenter defaultCenter] addObserverForName:NSBundleDidLoadNotification object:bundle queue:nil usingBlock:^(NSNotification *) {

    // Traverse each Boost test case in order to dynamically create an XCTestCase subclass for each Boost test suite
    // and a test method on it for each test case within that suite.

    auto registerer = xctest_dynamic_registerer();
    traverse_test_tree(framework::master_test_suite().p_id, registerer, true);
  }];
}

@end

/**
 * A category on the XCTestCase class to allow discovering of all the dynamically generated Boost test cases
 *
 * This approach was taken from https://github.com/mattstevens/xcode-googletest/
 */
@implementation XCTestCase(Boost2XCTest)

  /**
   * Implementation of +[XCTestCase testInvocations] that returns an array of test
   * invocations for each test method in the class.
   *
   * This differs from the standard implementation of testInvocations, which only
   * adds methods with a prefix of "test".
   */
  + (NSArray *)testInvocations
{
  NSMutableArray *invocations = [NSMutableArray array];

  unsigned int methodCount = 0;
  Method *methods = class_copyMethodList([self class], &methodCount);

  for (unsigned int i = 0; i < methodCount; i++) {
    SEL sel = method_getName(methods[i]);
    NSMethodSignature *sig = [self instanceMethodSignatureForSelector:sel];
    NSInvocation *invocation = [NSInvocation invocationWithMethodSignature:sig];
    [invocation setSelector:sel];
    [invocations addObject:invocation];
  }

  free(methods);

  return invocations;
}

@end
