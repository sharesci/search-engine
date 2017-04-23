import { Component } from '@angular/core';
import { AuthenticationService } from './services/authentication.service.js'
import { AccountService } from './services/account.service.js'
import { AppConfig } from './app.config.js'


@Component({
  selector: 'ss-app',
  template: `<div>
                <ss-navbar></ss-navbar>
                <router-outlet></router-outlet>
              </div>`,
  providers: [AuthenticationService, AccountService, AppConfig]
})

export class AppComponent {
  name = 'ShareSci';
}
