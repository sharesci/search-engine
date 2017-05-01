import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';
import { HttpModule } from '@angular/http';
import { FileSelectDirective, FileDropDirective } from 'ng2-file-upload';

import { AppComponent } from './app.component.js';
import { LoginComponent } from './components/login/login.component.js';
import { NavbarComponent } from './components/navbar/navbar.component.js'
import { HomeComponent } from './components/home/home.component.js'
import { CreateAccountComponent } from './components/create-account/create-account.component.js'
import { SearchResultComponent } from './components/search-result/search-result.component.js'
import { ProfileComponent } from './components/profile/profile.component.js'
import { ArticleComponent } from './components/article/article.component.js'
import { ArticleUploadComponent } from './components/article-upload/article-upload.component.js'

const appRoutes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: 'home', component: HomeComponent },
  { path: 'createaccount', component: CreateAccountComponent },
  { path: 'searchresult', component: SearchResultComponent },
  { path: 'profile/:username', component: ProfileComponent },
  { path: 'article/:id', component: ArticleComponent },
  { path: 'article/upload/:username', component: ArticleUploadComponent },
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  { path: '**', redirectTo: 'home', pathMatch: 'full' }
];

@NgModule({
  imports: [
    BrowserModule,
    FormsModule,
    HttpModule,
    RouterModule.forRoot(appRoutes, { useHash: true })
  ],
  declarations: [ AppComponent, LoginComponent, NavbarComponent, HomeComponent,
                  CreateAccountComponent, SearchResultComponent, ProfileComponent,
                  ArticleComponent, ArticleUploadComponent,
                  FileSelectDirective, FileDropDirective ],
  bootstrap: [AppComponent]
})
export class AppModule { }
